[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5258003.svg)](https://doi.org/10.5281/zenodo.5258003)
[![CI](https://github.com/TopEFT/topcoffea/actions/workflows/main.yml/badge.svg)](https://github.com/TopEFT/topcoffea/actions/workflows/main.yml)
[![Coffea-casa](https://img.shields.io/badge/launch-Coffea--casa-green)](https://cmsaf-jh.unl.edu/hub/spawn)
[![codecov](https://codecov.io/gh/TopEFT/topcoffea/branch/codecov/graph/badge.svg?token=U2DMI1C22F)](https://codecov.io/gh/TopEFT/topcoffea)

# topcoffea
Top quark analyses using the Coffea framework

### Contents
- `analysis`:
   Subfolders with different analyses: creating histograms, applying selections...
   Also including plotter scripts and/or jupyter files

- `topcoffea/cfg`:
  Configuration files (lists of samples, cross sections...)

- `topcoffea/data`:
  External inputs used in the analysis: scale factors, corrections...
  
- `topcoffea/json`:
   JSON files containing the lists of root files for each sample 

- `topcoffea/modules`:
  Auxiliar python modules and scripts

- `topcoffea/plotter`:
  Tools to produce stack plots and other plots

- `setup.py`: File for installing the `topcoffea` package

### Set up the environment 
First, download and install conda:
```
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > conda-install.sh
bash conda-install.sh
```
Next, run `unset PYTHONPATH` to avoid conflicts. Then run the following commands to set up the conda environment:    
```
conda env create -f environment.yml
conda activate coffea-env
```

### How to start
- This directory is set up to be installed as a python package. To install, activate your conda environment, then run this command from the top level `topcoffea` directory:
```
pip install -e .
```
The `-e` option installs the project in editable mode (i.e. setuptools "develop mode"). If you wish to uninstall the package, you can do so by running `pip uninstall topcoffea`.
- Next, set up the config file you want to use in the `topcoffea/cfg` directory. This config file should point to the JSON files for the samples that that you would like to process. There are examples in the `topcoffea/cfg` directory.
- Lastly, `cd` into `analysis/topEFT` and run the `run.py` script, passing it the path to your config: 
```
python run.py ../../topcoffea/cfg/your_cfg.cfg
```


### To run the WQ version of `run.py`:

To run with the work-queue executor, use the `work_queue_run.py` script instead of the `run.py` script. Please note that `work_queue_run.py` must be run from the directory it is located in, since the `extra-input-files` option of `executor_args` assumes the extra input will be in the current working directory. So from `analysis/topEFT`, you would run:
```
python work_queue_run.py ../../topcoffea/cfg/your_cfg.cfg
```
Next, submit some workers. Please note that the workers must be submitted from the same environment that you are running the run script from (so this will usually mean you want to activate the env in another terminal, and run the `condor_submit_workers` command from there. Here is an example `condor_submit_workers` command (remembering to activate the env prior to running the command):
```
conda activate coffea-env
condor_submit_workers -M ${USER}-workqueue-coffea -t 900 --cores 12 --memory 48000 --disk 100000 10
```
The workers will terminate themselves after 15 minutes of inactivity.


### How to fit the results
#### CMSSW
Install CMSSW_10_2_13 ***OUTSIDE OF YOUR TOPCOFFEA DIR AND NOT IN CONDA***
```
export SCRAM_ARCH=slc7_amd64_gcc700
scram project CMSSW CMSSW_10_2_13
cd CMSSW_10_2_13/src
scram b -j8
```

#### Set up Repo
This package is designed to be used with the CombineHarvester fork. Install within the same CMSSW release. See https://github.com/cms-analysis/CombineHarvester

##### Combine
Currently working with tag `v8.2.0`:

```
git clone git@github.com:cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit/
git checkout v8.2.0
cd -
scram b -j8
```

Otherwise, this package should be compatible with most CMSSW releases. It still requires the HiggsCombineTool package though. See https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/wiki/gettingstarted#for-end-users-that-dont-need-to-commit-or-do-any-development

##### EFTFit
```
cd $CMSSW_BASE/src/
git clone git@github.com:cms-govner/EFTFit.git EFTFit
scram b -j8
```

##### CombineHarvester
```
git clone git@github.com:cms-analysis/CombineHarvester.git
scram b -j8
```
This might case errors, but you can safely ignore them.

#### Fitting
##### In TopCoffea
- ROOT is required, so install it first iwth `conda install -c conda-forge root`
- Run `python analysis/topEFT/datacard_maker.py` (see `analysis/topEFT/README.md` for details)
##### In CMSSW
- Enter `CMSSW_10_2_13/src/EFTFit/Fitter/test` (wherever you have it installed) and run `cmsenv` to initialize CMSSW
- Copy all .txt and .root files created by `python analysis/topEFT/datacard_maker.py` (in the `histos` directory of your TopCoffea ananlyzer)
- Run `combineCards.py ttx_multileptons-* > combinedcard.txt` to merge them all into one txt file. **DO NOT** merge multiple variables!
- Run `text2workspace.py combinedcard.txt -o wps.root -P EFTFit.Fitter.AnomalousCouplingEFTNegative:analiticAnomalousCouplingEFTNegative --X-allow-no-background` to generate the workspace file
    - Specify a subset of WCs using e.g. `--PO cpt,ctp,cptb,cQlMi,cQl3i,ctlTi,ctli,cbW,cpQM,cpQ3,ctei,cQei,ctW,ctlSi,ctZ,ctG`
- Run combine with our EFTFit tools
  - Example:
```
python -i ../scripts/EFTFitter.py
fitter.batch1DScanEFT(basename='.081921.njet.ptbl.Float', batch='condor', workspace='wps.root')
```
  - Once all jobs are finished run `fitter.batchRetrieve1DScansEFT(basename='.081921.njet.ptbl.Float', batch='condor')` (again inside `python -i ../scripts/EFTFitter.py`) to collect them in the `EFTFit/Fitter/fit_files` folder
  - To make simple 1D plots, use:
```
python -i ../scripts/EFTPlotter.py
plotter.BatchLLPlot1DEFT(basename='.081121.njet.16wc.Float')
```
  - To make comparison plots (e.g. `njets` vs. `njets+ptbl`)
```
python -i ../scripts/EFTPlotter.py
plotter.BestScanPlot(basename_float='.081721.njet.Float', basename_freeze='.081821.njet.ptbl.Float', filename='_float_njet_ptbl', titles=['N_{jet} prof.', 'N_{jet}+p_{T}(b+l) prof.'], printFOM=True)
```
