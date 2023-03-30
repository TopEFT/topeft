[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5258003.svg)](https://doi.org/10.5281/zenodo.5258002)
[![CI](https://github.com/TopEFT/topcoffea/actions/workflows/main.yml/badge.svg)](https://github.com/TopEFT/topcoffea/actions/workflows/main.yml)
[![Coffea-casa](https://img.shields.io/badge/launch-Coffea--casa-green)](https://cmsaf-jh.unl.edu/hub/spawn)
[![codecov](https://codecov.io/gh/TopEFT/topcoffea/branch/master/graph/badge.svg?token=U2DMI1C22F)](https://codecov.io/gh/TopEFT/topcoffea)

# topcoffea
Top quark analyses using the Coffea framework

## Repository contents
- `analysis`:
   Subfolders with different analyses: creating histograms, applying selections...
   Also including plotter scripts and/or jupyter files

- `tests`:
   Scripts for testing the code with `pytest`. For additional details, please see the [README](https://github.com/TopEFT/topcoffea/blob/master/tests/README.md) in the `tests` directory.

- `topcoffea/cfg`:
  Configuration files (lists of samples, cross sections...)

- `topcoffea/data`:
  External inputs used in the analysis: scale factors, corrections...
  
- `topcoffea/json`:
   JSON files containing the lists of root files for each sample 

- `topcoffea/modules`:
  Auxiliar python modules and scripts

- `setup.py`: File for installing the `topcoffea` package


## Getting started

### Clone the repository
First, clone the repository:
```
git clone https://github.com/TopEFT/topcoffea.git
```

### Set up the environment 
Download and install conda:
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > conda-install.sh
bash conda-install.sh
```
Next, run `unset PYTHONPATH` to avoid conflicts. Then run the following commands to set up the conda environment (note that `environment.yml` is a file that is a part of the `topcoffea` repository, so you should `cd` into `topcoffea` before running the command):
```
conda env create -f environment.yml
conda activate coffea-env
```

### Install the topcoffea package
- This directory is set up to be installed as a python package. To install, activate your conda environment, then run this command from the top level `topcoffea` directory:
```
pip install -e .
```
The `-e` option installs the project in editable mode (i.e. setuptools "develop mode"). If you wish to uninstall the package, you can do so by running `pip uninstall topcoffea`.


### To run an example job with one of the two executors

First, set up the config file you want to use in the `topcoffea/cfg` directory. This config file should point to the JSON files for the samples that that you would like to process. There are examples in the `topcoffea/cfg` directory.

#### To run with the default workqueue executor:
- `cd` into `analysis/topEFT` and run the `run_topeft.py` script, passing it the path to your config:
```
python run_topeft.py ../../topcoffea/cfg/your_cfg.cfg
```
Next, submit some workers. Please note that the workers must be submitted from the same environment that you are running the run script from (so this will usually mean you want to activate the env in another terminal, and run the `condor_submit_workers` command from there. Here is an example `condor_submit_workers` command (remembering to activate the env prior to running the command):
```
conda activate coffea-env
condor_submit_workers -M ${USER}-workqueue-coffea -t 900 --cores 12 --memory 48000 --disk 100000 10
```
The workers will terminate themselves after 15 minutes of inactivity.

#### To run with the non-default futures executor:
- In the same `analysis/topEFT` directory, simply run:
```
python run_topeft.py -x futures ../../topcoffea/cfg/your_cfg.cfg
```

### How to contribute

If you would like to push changes to the TopCoffea repo, please push your new branch using `git push -u origin branch_name` where `origin` is the remote name for our repo, and `branch_name` is the name you would like to use (usually the same name in your local development area, but it doesn't have to be). After that, go the GitHub repo and open a PR. If you are developing on a fork, the CodeCov CI will fail. If possible, try to develope on the main repo instead.

__NOTE:__ If your branch gets out of date as other PRs are merged into the master branch, please run:
```bash
git fetch origin
git pull origin master
```
Depending on the changes, you might need to fix any conflicts, and then push these changes to your PR.

If your branch changes anything related to the yields, please run:
```bash
cd analysis/topEFT/
sh remake_ci_ref_yields.sh
sh remake_ci_ref_datacard.sh
```
The first script remakes the reference `json` file for the yields, and the second remakes the reference `txt` file for the datacar maker. If you are _certian_ these change are correct, commit and push them to the PR.

#### Installing pytest locally
To install `pytest` for local testing, run:
```bash
conda install -c conda-forge pytest pytest-cov
```
where `pytest-cov` is only used if you want to locally check the code coverage.

#### Running pytest locally

The `pytest` commands are run automatically in the CI. If you would like to run them locally, you can simply run:
```bash
pytest
```
from the main topcoffea directory. This will run _all_ the tests, which will take ~20 minutes. To run a subset, use e.g.:
```bash
pytest -k test_futures
```
where `test_futures` is the file/test you would like to run (check the `tests` directory for all the available tests, or write your own and push it!). If you would also like to see how the coverage changes, you can add `--cov=./ --cov-report=html` to `pytest` commands. This will create an `html` directory that you can then copy to any folder which you have web access to (e.g. `~/www/` on Earth) For a better printout of what passed and failed, add `-rP` to the `pytest` commands.


### Further reading 

* For more details about work queue, please see README_WORKQUEUE.md
* For more details about how to fit the results, please see README_FITTING.md


## To reproduce the TOP-22-006 histograms and datacards

Use the [v0.5 tag](https://github.com/TopEFT/topcoffea/releases/tag/v0.5)

1. Run the processor to obtain the histograms (from the skimmed naod files). Use the `fullR2_run.sh` script in the `analysis/topEFT` directory.
    ```
    time source fullR2_run.sh
    ```

2. Run the datacard maker to obtain the cards and templates (from the pickled histogram file produced in Step 1, be sure to use the version with the nonprompt estimation, i.e. the one with `_np` appended to the name you specified for the `OUT_NAME` in `fullR2_run.sh`).
    ```
    time python make_cards.py /path/to/your/examplename_np.pkl.gz -C --do-nuisance --var-lst lj0pt ptz -d /scratch365/you/somedir --unblind --do-mc-stat
    ```

3. Run the post-processing checks on the cards to look for any unexpected errors in the condor logs and to grab the right set of ptz and lj0pt templates and cards used in TOP-22-006. The script will copy the relevant cards/templates to a directory it makes (inside of the directory you pass that points to the cards and templates made in step 2). This is the directory that can be copied to wherever you plan to run the `combine` steps (e.g. PSI).
    ```
    time python datacards_post_processing.py /scratch365/you/somedir -c -s
    ```

4. Proceed with the "Steps for reproducing the "official" TOP-22-006 workspace" steps listed in the EFTFit Readme [here](https://github.com/TopEFT/EFTFit#steps-for-reproducing-the-official-top-22-006-workspace).


