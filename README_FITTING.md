# How to fit the results

The first step is to produce the datacard text files and root files that combine will use, and this step takes place within `topcoffea`.  The next step is to run combine, which takes place inside of a CMSSW release, outside of `topcoffea`.

## Creating the datacards

The first step is to produce the datacard text files and root files that combine will use. This step takes place within `topcoffea`. Run the `make_cards.py` script to produce the data cards.

Notes for ND users: When running steps involving condor, if you want to write to your `afs` area you will need to make sure that the permissions in your `afs` are are set properly, as outlined in the ND T3 [documentation](https://docs.crc.nd.edu/resources/NDCMS/ndcms.html#setting-up-environment). It is easier to write to somewhere besides your `afs` area, e.g. your area is `/scratch365`.

Example of running the `make_cards.py` script:
```
python make_cards.py path/to/your.pkl.gz -C --do-nuisance --var-lst lj0pt ptz -d /scratch365/yourusername/some/dir
```

## Running combine

:warning: The EFT basis rotation does not compile correctly on `glados`. Please make your workspace on a tested machine like `lxplus`. The root limit scans can still be done on `glados`.

The next step is to run combine. This takes place inside of a CMSSW release, outside of `topcoffea`. See the [EFTFit](https://github.com/TopEFT/EFTFit) repo for instructions.
