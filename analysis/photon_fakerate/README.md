###Setting up
The setups are exactly the same as the one for running the main analysis processor.

###What do each of the scripts do?
- `derive_nonprompt_fakerate.py`: This script is used to derive the nonprompt photon fakerate (FR). The same FR is good for the ABCD and LRCD (validation) regions. The FR derivation takes all the data and MC samples.

- `derive_nonprompt_kmc.py`: This script is used to derive the kMC factors for the ABCD region. The kMC derivation only requires ttbar MC samples.

- `derive_nonprompt_kmc_LRCD.py`: This script is used to derive the kMC factors for the LRCD validation region. The kMC validation only requires ttbar MC samples.

###Important notes
- Currently we do not include systematic uncertainties when derive the fake rates and kMC factors.

###Commands:

1. Nonprompt fakerate derivation: 
```
time python run_derive_nonprompt_fakerate.py <data_cfg>,<mc_cfg> --hist-list nonprompt -s 50000 -o <output_pkl_file_name>
```

2. kMC derivation:
```
time python run_derive_nonprompt_kmc.py <ttbar_mc_cfg> --hist-list nonprompt -s 50000 -o <output_pkl_file_name>
```

3. kMC validation derivation:
```
time python run_derive_nonprompt_kmc_LRCD.py <ttbar_mc_cfg> --hist-list nonprompt -s 50000 -o <output_pkl_file_name>
```

###Extracting FR and kMC
Once the pkl files are ready, one can use `calculate_FR.py`, `calculate_kMC.py`, and `calculate_kMC_LRDC.py` scripts to extract the fake rates and save them as numpy (.npz) files. Once these files are ready, they should be saved inside `topeft/data/photon_fakerates/` or `topeft/data/photon_kmc/` or `topeft/data/photon_kmc_validation` dirs.

###Running non-prompt validation test
TODO: add instructions
