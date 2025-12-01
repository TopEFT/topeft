[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5258003.svg)](https://doi.org/10.5281/zenodo.5258002)
[![CI](https://github.com/TopEFT/topcoffea/actions/workflows/main.yml/badge.svg)](https://github.com/TopEFT/topeft/actions/workflows/main.yml)
[![Coffea-casa](https://img.shields.io/badge/launch-Coffea--casa-green)](https://cmsaf-jh.unl.edu/hub/spawn)
[![codecov](https://codecov.io/gh/TopEFT/topcoffea/branch/master/graph/badge.svg?token=U2DMI1C22F)](https://codecov.io/gh/TopEFT/topcoffea)

# topeft
Top quark EFT analyses using the Coffea framework

> **Workflow + YAML overview**: Start with the new [workflow and YAML hub](docs/workflow_and_yaml_hub.md) to see how the Run 2 presets, executors (futures or TaskVine), and metadata files fit together before diving into the rest of the docs.

## Repository contents
The `topeft/topeft` directory is set up to be installed as a pip installable package.
- `topeft/topeft`: A package containing modules and files that will be installed into the environment.
- `pyproject.toml`: PEP 517 metadata describing the package, dependencies, and bundled data.
- `topeft/setup.py`: Minimal shim for invoking the package build
- `topeft/analysis`: Subfolders with different analyses or studies.
- `topeft/tests`: Scripts for testing the code with `pytest`. For additional details, please see the [README](https://github.com/TopEFT/topeft/blob/master/tests/README.md) in the `tests` directory.
- `topeft/input_samples`: Configuration files that point to root files to process.

### Legacy HistEFT support
The repository now expects the [`topcoffea`](https://github.com/TopEFT/topcoffea) package to be installed in the active environment instead of shipping a partial vendor copy.  The upstream project still exposes the legacy `HistEFT` helpers via `topcoffea.modules.HistEFT`, so existing imports keep working once the dependency is installed.
CI now includes a smoke test that asserts `import topcoffea` resolves outside the `topeft` checkout.  Remove any stray `topcoffea/` directories under this repository and reinstall the sibling package (for example via `pip install -e ../topcoffea`) if that guard triggers.

## Getting started

New to the workflow? The [Run and plot quickstart](docs/run_and_plot_quickstart.md) walks through creating the environment, running `run_analysis.py` with both the futures and TaskVine executors, and turning the tuple-keyed histogram pickle into plots.

### Setting up
If conda is not already available, download and install it:
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > conda-install.sh
bash conda-install.sh
```
The topeft directory is set up to be installed as a python package. First clone the repository as shown, then run the following commands to set up the environment (note that `environment.yml` is a file that is a part of the `topeft` repository, so you should `cd` into `topeft` before running the command).  The refreshed workflow standardizes on the Coffea 2025.7.3 toolchain captured in the `coffea2025` Conda environment shared with the [`ttbarEFT`](https://github.com/TopEFT/ttbarEFT) analysis:
```
git clone https://github.com/TopEFT/topeft.git
cd topeft
unset PYTHONPATH # To avoid conflicts.
conda env create -f environment.yml
conda activate coffea2025
pip install -e .
```
The commands above provision the refreshed `coffea2025` Conda environment, which tracks the Coffea 2025.7.3 base release used throughout the project.  Installing `topeft` with `pip install -e .` is now the expected workflow so that local developments are automatically folded into the packaged environment.

The Conda specification now also pins `hist=2.9.*` and `boost-histogram>=1.4` to mirror the histogram stack required by the sibling `topcoffea` checkout.  Keep the NumPy and pandas constraints intact when updating the environment so that ABI compatibility stays aligned across both projects.

To install both sibling repositories in one step, either request the new optional dependency group or run the helper script:

```
pip install -e .[topcoffea]
# or
./scripts/dev_install.sh
```

The first form pulls `topcoffea` from a sibling checkout at `../topcoffea` while the second mirrors the original command (`pip install -e ../topcoffea -e .`) for contributors who prefer an explicit, reproducible installer.

If your workstation already provides a `conda` executable, the helper commands above will reuse it.  On minimal runners or fresh containers where only `micromamba` is present, the `topeft` package now exposes a lightweight `conda` shim that forwards invocations to `micromamba` so `python -m topcoffea.modules.remote_environment` continues to work without a full Conda installation.

Upgrading from an older checkout?  Reuse the same directory and run `conda env update -f environment.yml --prune` before activating the environment so the existing `coffea2025` install picks up the Coffea 2025.7.3 pins and removes stale dependencies.  If the solver prompts for an update, accept it or run `conda update -n base -c conda-forge conda` first to keep in sync with conda-forge builds.

To catch configuration drift, CI hashes `environment.yml` and compares it to the upstream `ttbarEFT` coffea2025 specification (see `tests/test_environment_hash.py`).  When the upstream environment changes, update this repository's `environment.yml` and refresh the expected digest in that test so the guard continues to pass.

The `-e` option installs the project in editable mode (i.e. setuptools "develop mode"). If you wish to uninstall the package, you can do so by running `pip uninstall topeft`.
The [`topcoffea`](https://github.com/TopEFT/topcoffea) package upon which this analysis depends is not yet available on `PyPI`, so we need to clone the `topcoffea` repo and install it ourselves.  Keep the checkout next to `topeft` (for example `$HOME/workspace/topcoffea` beside `$HOME/workspace/topeft`) so paths such as `analysis/topeft_run2/../../topcoffea/cfg/...` resolve to the sibling repository when invoking older scripts.
```
cd /your/favorite/directory
git clone https://github.com/TopEFT/topcoffea.git
cd topcoffea
git switch ch_update_calcoffea
pip install -e .
cd ../topeft
python -m topcoffea.modules.remote_environment
python -c "import topcoffea"
```
Keeping both repositories installed in editable mode ensures the remote-packaging helper inspects the current checkouts (and fails fast if there are unstaged edits).  **Always switch the sibling checkout to `ch_update_calcoffea` (or the matching release tag) before running `pip install -e ../topcoffea` _and_ when invoking `python -m topcoffea.modules.remote_environment`.**  Packaging the tarball from the same branch keeps the Conda + pip stack aligned with the processors.  Analysts relying on a detached tag can set `TOPCOFFEA_BRANCH=<tag>` before running `topeft` so the guard recognises the pinned reference.  All CLI entry points now validate that the resolved branch matches the expected `ch_update_calcoffea` baseline (using `.git/HEAD` when available or the `TOPCOFFEA_BRANCH` override), raising a clear error if the sibling checkout drifts.

The `python -m topcoffea.modules.remote_environment` command calls into the updated `remote_environment.get_environment()` logic, which assembles a fresh TaskVine-ready tarball under `topeft-envs/`, captures the pinned Conda + pip stack, and returns the archive path that the workflow passes through the `environment_file` executor argument.  When the inputs match a previously cached build, it reuses the existing archive instead of rebuilding, but any change to the dependency specification or editable sources will trigger a refresh.  The final `python -c "import topcoffea"` line mirrors the CI smoke test and confirms the namespace import succeeds for downstream tooling.

When editing dependencies (for example bumping NumPy/Pandas pins), recreate the `coffea2025` environment and force a new TaskVine tarball so the cached archive does not hold on to an outdated ABI mix.  Activate the refreshed environment for both local futures runs and TaskVine submissions so the manager and workers stay in sync.

#### Packaged TaskVine environment cache

The cached tarballs live in `topeft-envs/` and are named after the Conda + pip specification combined with the Git commits of editable packages.  Each invocation of `python -m topcoffea.modules.remote_environment` prints the active path and automatically rebuilds the archive when it detects new commits or unstaged edits in `topeft` or `topcoffea`.  If you need to force a rebuild (for example after cleaning a branch or rebasing), either remove the cached file or run `python -c "from topcoffea.modules.remote_environment import get_environment; print(get_environment(force=True))"`.  The resulting tarball is exactly what `processor.TaskVineExecutor` sends to remote workers through the `environment_file` parameter.

Now all of the dependencies have been installed and the `topeft` repository is ready to be used. The packaged environment targets the Coffea 2025.7.3 release and can be re-generated at any time with `python -m topcoffea.modules.remote_environment`. The next time you want to use the project, activate the environment via `conda activate coffea2025`.


### TaskVine distributed execution

For a narrated walkthrough that ties environment preparation, tarball packaging, and worker submission together, see the [TaskVine workflow quickstart](docs/taskvine_workflow.md). The packaged archive produced by `python -m topcoffea.modules.remote_environment` is uploaded automatically when runs target the TaskVine executor. Launching a distributed job therefore becomes:

1. Enable TaskVine in the YAML profile before launching the run.  Open
   `analysis/topeft_run2/configs/fullR2_run.yml` and set the profile you plan
   to execute to use the TaskVine backend (the preset defaults to
   `executor: futures` for local smoke tests):

   ```yaml
   profiles:
     cr:
       # ...existing options...
       executor: taskvine
   ```

   (When you prefer to keep both backends handy, copy the file to a new name
   such as `fullR2_run_taskvine.yml` and adjust the `executor` there.)

2. Configure the workflow with TaskVine enabled (for example, selecting the
   Run 2 control preset).  Setting `executor: taskvine` in the YAML tells
   `run_analysis.py` to launch `coffea.processor.TaskVineExecutor`, which
   automatically streams tasks to the TaskVine manager while handing workers
   the packaged environment via `environment_file`:

   ```bash
   python run_analysis.py \
       ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
       --options configs/fullR2_run.yml
   ```

   The `analysis/topeft_run2/full_run.sh` wrapper automates this setup when you
   want a single entry point. It activates the shared Conda environment when
   available, refreshes the cached TaskVine tarball via
   `python -m topcoffea.modules.remote_environment`, and forwards control- or
   signal-region selections to `run_analysis.py` while keeping the TaskVine
   executor defaults (manager name `${USER}-taskvine-coffea`, 5-tuple histogram
   pickle outputs). Example end-to-end invocation:

   ```bash
   cd analysis/topeft_run2
   ./full_run.sh --sr -y 2022 2022EE --outdir histos/run3_taskvine --tag nightly
   ```

3. Start one or more workers that match the manager name advertised by the run. A minimal local worker looks like:

   ```bash
   vine_worker --cores 1 --memory 8000 --disk 8000 -M ${USER}-taskvine-coffea
   ```

   When scaling out through TaskVine's submission helpers, prefer launching a worker pool with the packaged environment already attached:

   ```bash
   vine_submit_workers --cores 4 --memory 16000 --disk 16000 \
       --python-env "$(python -m topcoffea.modules.remote_environment)" \
       -M ${USER}-taskvine-coffea 10
   ```

   The `remote_environment.get_environment()` helper returns the same tarball path that the workflow passes through the `environment_file` executor argument; pre-loading it with `--python-env` avoids transferring the archive every time a new worker connects. If `--python-env` is omitted, the manager still hands the environment off automatically via `environment_file`, matching the behaviour of older Work Queue deployments.

   For HTCondor-backed pools, `condor_submit_workers` and other submission helpers ship with TaskVine; point them at the same manager string and pass `--python-env` when the helper supports it. Refer to the [remote environment maintenance guide](docs/environment_packaging.md) for details on rebuilding the tarball whenever dependencies change.

   The `analysis/topeft_run2/full_run.sh` wrapper now supports both TaskVine
   (default) and local futures executions. Add `--executor futures` to switch to
   the single-node path and `--dry-run` to print the resolved command without
   launching Python. The same entrypoint accepts `--samples` overrides for quick
   JSON-based smoke tests:

   ```bash
   cd analysis/topeft_run2
   ./full_run.sh --sr -y UL17 --executor futures --samples \
       ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
       --outdir histos/local_debug --tag quickstart --dry-run
   ```

   Work Queue has been retired from the workflow helpers. A condensed record of the historic instructions is preserved in [README_WORKQUEUE.md](README_WORKQUEUE.md) for teams pinned to older releases.

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
isolated.  Quickstart helpers can still consume the clone directly via their own
`--metadata` flag, but `run_analysis.py` now resolves metadata solely through
the scenario registry or YAML options files.  Add a `metadata:` entry to the
profile you launch with `--options` so the override lives alongside the rest of
your configuration knobs.  A minimal example profile looks like:

```yaml
# analysis/topeft_run2/configs/fullR2_run_myteam.yml
metadata: configs/metadata_myteam.yml
jsonFiles:
  - ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json
scenarios:
  - TOP_22_006
```

Run the custom configuration with:

```bash
python run_analysis.py --options analysis/topeft_run2/configs/fullR2_run_myteam.yml
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
can be reused across distributed resources.  A short legacy note is available in
[README_WORKQUEUE.md](README_WORKQUEUE.md) for teams referencing the historical
Work Queue setup.

For local futures runs the CLI now exposes dedicated knobs for staging and
recovery: ``--futures-prefetch`` limits the number of ROOT files staged per
sample, ``--futures-retries``/``--futures-retry-wait`` control automatic
resubmission, and ``--futures-status`` or ``--futures-tail-timeout`` let you
adjust progress logging.  YAML profiles accept the same keys with underscores.
The [analysis/topeft_run2/README.md](analysis/topeft_run2/README.md) guide
collects practical recipes for mixing these options when debugging before
handing jobs off to TaskVine.


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

1. Run the processor to obtain the histograms (from the skimmed naod files).
   The preserved presets live in ``analysis/topeft_run2/configs/fullR2_run.yml``;
   launch them through the unified wrapper:

    ```bash
    cd analysis/topeft_run2
    ./full_run.sh --cr -y run2 --options configs/fullR2_run.yml --outdir histos/run2_ref --tag top22006
    ```

2. Run the datacard maker to obtain the cards and templates from SM (from the pickled histogram file produced in Step 1, be sure to use the version with the nonprompt estimation, i.e. the one with `_np` appended to the name you specified with the ``--tag``/``--outdir`` pair in ``full_run.sh``). This step would also produce scalings-preselect.json file which the later version is necessary for IM workspace making. Note that command option `--wc-scalings` is not mandatory but to enforce the ordering of wcs in scalings. Add command `-A` to include all EFT templates in datacards for previous AAC model. Add option `-C` to run on condor.
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

