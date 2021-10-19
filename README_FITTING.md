# How to fit the results

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
