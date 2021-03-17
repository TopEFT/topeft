# topcoffea
Top quark analyses using the Coffea framework

This is a developing effort... no complete instructions or test yet!


### Contents
- `analysis`:
   Subfolders with different analyses: creating histograms, applying selections...
   Also including plotter scripts and/or jupyter files

- `cfg`:
  Configuration files (lists of samples, cross sections...)

- `data`:
  External inputs used in the analysis: scale factors, corrections...

- `topcoffea/modules`:
  Auxiliar python modules and scripts

- `topcoffea/plotter`:
  Tools to produce stack plots and other plots

- `setup.py`: File for installing the `topcoffea` package

### Set up the environment 
If necessary, first run `unset PYTHONPATH` to avoid conflicts. Then run the following commands to set up the conda environment:    
```
conda env create -f environment.yml
conda activate coffea-env
```

### How to start
- This directory is set up to be installed as a python package. To install, run `pip install -e .` from the top level directory. The `-e` option installs the project in editable mode (i.e. setuptools "develop mode"). If you wish to uninstall the package, you can do so by running `pip uninstall topcoffea`.
- Set up the cfg file you want to use in the `cfg` directory.
- Run `analysis/topEFT/run.py cfg/your_cfg.cfg`.
