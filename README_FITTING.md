# How to fit the results

The first step is to produce the datacard text files and root files that combine will use, and this step takes place within `topcoffea`.  The next step is to run combine, which takes place inside of a CMSSW release, outside of `topcoffea`.

Notes for ND users: When running steps involving condor, you will need to make sure that the permissions in your `afs` are are set properly, as outlined in the ND T3 [documentation](https://docs.crc.nd.edu/resources/NDCMS/ndcms.html#setting-up-environment). Specifically:
* For running the datacard maker, you'll need to make sure that the permissions for your `histos` directory and your `condor/log/` directory are `rlidwka` for `nd_campus`, `system:administrators `, and `system:authuser`.
* For running combine, you'll need to make sure that the permissions for your `CMSSW_10_2_13/src/EFTFit/Fitter/test` directory are `rlidwka` for `nd_campus`, `system:administrators `, and `system:authuser`.
* Note that you can check the current permissions for your your current directory with `fs la .`.

## Creating the datacards

The first step is to produce the datacard text files and root files that combine will use. This step takes place within `topcoffea`.
- ROOT is required, so install it first iwth `conda install -c conda-forge root_base`
- Run `python analysis/topEFT/datacard_maker.py` (see `analysis/topEFT/README.md` for details)

## Running combine

 The next step is to run combine. This takes place inside of a CMSSW release, outside of `topcoffea`.
 
 ### Setting up
 
  In order to run combine, you will need to get the appropriate CMSSW release and to clone several repositories.

#### Set up the CMSSW release
Install CMSSW_10_2_13 ***OUTSIDE OF YOUR TOPCOFFEA DIR AND NOT IN CONDA***
```
export SCRAM_ARCH=slc7_amd64_gcc700
scram project CMSSW CMSSW_10_2_13
cd CMSSW_10_2_13/src
scram b -j8
```

#### Get the Combine repository
Currently working with tag `v8.2.0`:

```
git clone git@github.com:cms-analysis/HiggsAnalysis-CombinedLimit.git HiggsAnalysis/CombinedLimit
cd HiggsAnalysis/CombinedLimit/
git checkout v8.2.0
cd -
scram b -j8
```

#### Get the EFTFit repository
```
cd $CMSSW_BASE/src/
git clone https://github.com/TopEFT/EFTFit.git EFTFit
scram b -j8
```

#### Get the CombineHarvester repository
This package is designed to be used with the CombineHarvester fork. This might cause errors when compiling, but you can safely ignore them.

```
git clone git@github.com:cms-analysis/CombineHarvester.git
scram b -j8
```


### Fitting

Now we can actually run combine to perform the fits.

#### Running the fits
- Make sure you have done a `cmsenv` inside of `CMSSW_10_2_13/src/` (wherever you have it installed)
- Enter `CMSSW_10_2_13/src/EFTFit/Fitter/test`
- Copy all .txt and .root files created by `python analysis/topEFT/datacard_maker.py` (in the `histos` directory of your toplevel topcoffea directory)
- Run `combineCards.py ttx_multileptons-*.txt > combinedcard.txt` to merge them all into one txt file. **DO NOT** merge multiple variables!
- NOTE: combine uses a lot of recursive function calls to create the workspace. When running with systematics, this can cause a segmentation fault. You must run `ulimit -s unlimited` once per session to avoid this.
- Run the following command to generate the workspace file:
    ```
    text2workspace.py combinedcard.txt -o wps.root -P EFTFit.Fitter.AnomalousCouplingEFTNegative:analiticAnomalousCouplingEFTNegative --X-allow-no-background
    ``` 
    You can Specify a subset of WCs using `--PO`, e.g.:
    ```
    text2workspace.py combinedcard.txt -o wps.root -P EFTFit.Fitter.AnomalousCouplingEFTNegative:analiticAnomalousCouplingEFTNegative --X-allow-no-background --PO cpt,ctp,cptb,cQlMi,cQl3i,ctlTi,ctli,cbW,cpQM,cpQ3,ctei,cQei,ctW,ctlSi,ctZ,ctG
    ```
- Run combine with our EFTFit tools
  - Example:
    ```
    python -i ../scripts/EFTFitter.py
    fitter.batch1DScanEFT(basename='.081921.njet.ptbl.Float', batch='condor', workspace='wps.root')
    ```
  - Once all jobs are finished, run the following (again inside `python -i ../scripts/EFTFitter.py`) to collect them in the `EFTFit/Fitter/fit_files` folder: 
    ```
    fitter.batchRetrieve1DScansEFT(basename='.081921.njet.ptbl.Float', batch='condor')
    ````

#### Plot making

To make simple 1D plots, use:
```
python -i ../scripts/EFTPlotter.py
plotter.BatchLLPlot1DEFT(basename='.081121.njet.16wc.Float')
```
To make comparison plots (e.g. `njets` vs. `njets+ptbl`):
```
python -i ../scripts/EFTPlotter.py
plotter.BestScanPlot(basename_float='.081721.njet.Float', basename_freeze='.081821.njet.ptbl.Float', filename='_float_njet_ptbl', titles=['N_{jet} prof.', 'N_{jet}+p_{T}(b+l) prof.'], printFOM=True)
```
