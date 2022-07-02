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

 The next step is to run combine. This takes place inside of a CMSSW release, outside of `topcoffea`. See the [EFTFit](https://github.com/TopEFT/EFTFit) repo for instructions.
