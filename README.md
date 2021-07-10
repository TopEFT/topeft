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
conda create --name topcoffea-env python=3.8.3
conda activate topcoffea-env
conda install -y -c conda-forge ndcctools conda-pack dill xrootd coffea
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
conda activate topcoffea-env
condor_submit_workers -M ${USER}-workqueue-coffea --cores 8 --memory 56000 --disk 100000 10
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
This package is designed to be used with the cms-govner CombineHarvester fork. Install within the same CMSSW release. See https://github.com/cms-govner/CombineHarvester

##### Combine
Currently working with tag `v8.2.0`:

```
git clone https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit/
git checkout v8.2.0
cd -
scram b -j8
```

Otherwise, this package should be compatible with most CMSSW releases. It still requires the HiggsCombineTool package though. See https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit/wiki/gettingstarted#for-end-users-that-dont-need-to-commit-or-do-any-development

##### CombineHarvester
```
git clone https://github.com/cms-govner/CombineHarvester.git CombineHarvester
scram b -j8
```
This might case errors, but you can safely ignore them.

##### EFTFit
```
cd $CMSSW_BASE/src/
git clone https://github.com/cms-govner/EFTFit.git EFTFit
scram b -j8
```

#### Fitting
##### In TopCoffea
- Run `python analysis/topEFT/datacard_maker.py` (see `analysis/topEFT/README.md` for details)
##### In CMSSW
- Enter `CMSSW_10_2_13/src/EFTFit/Fitter/test` (wherever you have it installed) and run `cmsenv` to initialize CMSSW
- Copy all .txt and .root files created by `python analysis/topEFT/datacard_maker.py` (in the `histos` directory of your TopCoffea ananlyzer)
- Run `combineCards.py ttx_multileptons-* > combinedcard.txt` to merge them all into one txt file
- Run `text2workspace.py combinedcard.txt -o wps.root -P EFTFit.Fitter.AnomalousCouplingEFTNegative:analiticAnomalousCouplingEFTNegative --X-allow-no-background` to generate the workspace file
- Run combine
  - Example `combineTool.py  wps.root -M MultiDimFit --algo grid -t -1 --setParameters  ctW=0,ctp=0,cpQM=0,ctli=0,cQei=0,ctZ=0,cQlMi=0,cQl3i=0,ctG=0,ctlTi=0,cbW=0,cpQ3=0,ctei=0,cpt=0,ctlSi=0,cptb=0,cQq13=0,cQq83=0,cQq11=0,ctq1=0,cQq81=0,ctq8=0,r=1 -P ctW --freezeParameters ctG,ctp,cpQM,ctli,cQei,ctZ,cQlMi,cQl3i,ctlTi,cbW,cpQ3,ctei,cpt,ctlSi,cptb,cQq13,cQq83,cQq11,ctq1,cQq81,ctq8,r --setParameterRanges ctW=-6,6 --trackParameters cQei --points 200 --job-mode condor --split-point 20`
