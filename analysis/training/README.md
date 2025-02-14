[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TopEFT/topeft/master?labpath=analysis%2Ftraining%2Fintro_coffea.hist.ipynb)<br>

## training
This directory contains example scripts that may be useful for learning to run coffea and topcoffea. 

### Past tutorials:

* Jun 2021: [2021 TopCoffea tutoral](https://indico.cern.ch/event/1047567/)
* Aug 2022: [2022 TopCoffea tutorial Session 1](https://indico.cern.ch/event/1188768/)
* Sep 2022: [2022 TopCoffea tutorial Session 2](https://indico.cern.ch/event/1189721/)
* Jan 2023: [2023 Advanced TopCoffea tutorial](https://indico.cern.ch/event/1228170/)


### Scripts:

* `simple_processor.py` and `simple_run.py`: A minimal example of a topcoffea processor. The processor can be run as follows.
    - Download a root file to the local directory, e.g.:
    ```
    wget -nc http://www.crc.nd.edu/~kmohrman/files/root_files/for_ci/ttHJet_UL17_R1B14_NAOD-00000_10194_NDSkim.root
    ```
    - Run the run script over a json that points to the root file you downloaded: 
    ```
    python simple_run.py ../../topcoffea/json/test_samples/UL17_private_ttH_for_CI.json
    ```

* `intro_coffea.hist.py` and `intro_coffea.hist.ipynb`: A very basic introduction to coffea histogram (deprecated). Introduces some important coffea methods used in filling, transforming, and plotting the hist object.
    - To run the jupyter notebook, one can use coffea casa facility (instructions to login to coffea-casa can be found [here](https://coffea-casa.readthedocs.io/en/latest/cc_user.html#access)). Then, one can clone this repository using [these instructions](https://coffea-casa.readthedocs.io/en/latest/cc_user.html#using-git). 
