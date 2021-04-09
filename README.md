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
conda create --name coffea-070-env python=3.8.3
conda activate coffea-070-env
pip install coffea==0.7.0
conda install -c conda-forge xrootd
```

### How to start
- This directory is set up to be installed as a python package. To install, run `pip install -e .` from the top level directory. The `-e` option installs the project in editable mode (i.e. setuptools "develop mode"). If you wish to uninstall the package, you can do so by running `pip uninstall topcoffea`.
- Set up the cfg file you want to use in the `topcoffea/cfg` directory.
- Run `analysis/topEFT/run.py topcoffea/cfg/your_cfg.cfg`.

### How to convert HistEFT to TH1EFT
- The ROOT files must be compiled (`root -q -b Utils/WCFit.h+` and `root -q -b Utils/TH1EFT.cc+`)
- Run `python analysis/topEFT/convert3lEFT.py` to perform conversion of MET
- See analysis/topEFT/convert3lEFT.py for more details
