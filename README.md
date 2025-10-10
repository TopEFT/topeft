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
conda activate coffea-env
pip install -e .
```
The `-e` option installs the project in editable mode (i.e. setuptools "develop mode"). If you wish to uninstall the package, you can do so by running `pip uninstall topcoffea`. 
The `topcoffea` package upon which this analysis also depends is not yet available on `PyPI`, so we need to clone the `topcoffea` repo and install it ourselves.
```
cd /your/favorite/directory
git clone https://github.com/TopEFT/topcoffea.git
cd topcoffea
pip install -e .  
```
Now all of the dependencies have been installed and the `topeft` repository is ready to be used. The next time you want to use it, all you have to do is to activate the environment via `conda activate coffea-env`. 


### To run an example job 

First `cd` into `analysis/topeft_run2` and run the `run_analysis.py` script, passing it the path to your config file or json file. In this example we'll process a single root file locally, using a json file that is already set up.
```
cd analysis/topeft_run2
wget -nc http://www.crc.nd.edu/~kmohrman/files/root_files/for_ci/ttHJet_UL17_R1B14_NAOD-00000_10194_NDSkim.root
python run_analysis.py ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json -x futures

```
Executable examples for both the CLI-only and YAML-metadata workflows are provided in `analysis/topeft_run2/examples/`:

```bash
./analysis/topeft_run2/examples/run_with_cli.sh
./analysis/topeft_run2/examples/run_with_yaml.sh
```
The YAML script uses `analysis/topeft_run2/examples/yaml_metadata_example.yaml` to supply configuration values that mirror the refactored options loader.
To make use of distributed resources, the `work queue` executor can be used. To use the work queue executor, just change the executor option to  `-x work_queue` and run the run script as before. Next, you will need to request some workers to execute the tasks on the distributed resources. Please note that the workers must be submitted from the same environment that you are running the run script from (so this will usually mean you want to activate the env in another terminal, and run the `condor_submit_workers` command from there. Here is an example `condor_submit_workers` command (remembering to activate the env prior to running the command):
```
conda activate coffea-env
condor_submit_workers -M ${USER}-work_queue-coffea -t 900 --cores 12 --memory 48000 --disk 100000 10
```
The workers will terminate themselves after 15 minutes of inactivity. More details on the work queue executor can be found [here](https://github.com/TopEFT/topeft/blob/master/README_WORKQUEUE.md).


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


