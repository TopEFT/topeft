# topcoffea
Top quark analyses using the Coffea framework

This is a developing effort... no complete instructions or test yet!


### Contents
- `analysis`:
   Subfolders with different analyses: creating histograms, applying selections...
   Also including plotter scripts and/or jupyter files

- `topcoffea/cfg`:
  Configuration files (lists of samples, cross sections...)

- `topcoffea/data`:
  External inputs used in the analysis: scale factors, corrections...

- `topcoffea/modules`:
  Auxiliar python modules and scripts

- `topcoffea/plotter`:
  Tools to produce stack plots and other plots

- `setup.py`: File for installing the `topcoffea` package

### Set up the environment 
If necessary, first run `unset PYTHONPATH` to avoid conflicts. Then run the following commands to set up the conda environment:    
```
conda create --name coffea-env python=3.8.3
conda activate coffea-env
conda install -y -c conda-forge coffea
```

### How to start
- This directory is set up to be installed as a python package. To install, run `pip install -e .` from the top level directory. The `-e` option installs the project in editable mode (i.e. setuptools "develop mode"). If you wish to uninstall the package, you can do so by running `pip uninstall topcoffea`.
- Set up the cfg file you want to use in the `topcoffea/cfg` directory.
- Run `analysis/topEFT/run.py topcoffea/cfg/your_cfg.cfg`.

### How to convert HistEFT to TH1EFT
- Run `source setupTH1EFT.sh` to download and compile ROOT files
- Run `python analysis/topEFT/convert3lEFT.py` to perform conversion of MET
- See `analysis/topEFT/convert3lEFT.py` for more details


### To run the WQ version of `run.py`:
Set up the environment (only needed once):
```
conda create --name topcoffea-env python=3.8.3 conda
conda activate topcoffea-env
conda install -y -c conda-forge ndcctools conda-pack dill xrootd coffea
pip install .
```

The next step is to run `work_queue_run.py`. Please note that this is still a work in progress, and some of the lines are still hard coded; specifically, `environment-file` and `wrapper` in `executor_args` are hardcoded, so please adjust them accordingly before you run. Also note that `work_queue_run.py` must be run from the directory it is located in, since the `extra-input-files` option of `executor_args` assumes the extra input will be in the current working directory. So from `topcoffea/analysis/topEFT`, you would run:
```
conda activate topcoffea-env
cd analysis/topEFT
python work_queue_run.py ../../topcoffea/cfg/your_cfg.cfg
```

Next, submit some workers, e.g.:
```
conda activate topcoffea-env
condor_submit_workers -M ${USER}-workqueue-coffea --cores 4 --memory 4000 --disk 2000 10
```

Workers terminate themselves after 15 minutes of inactivity.
