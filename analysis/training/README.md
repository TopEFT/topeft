## training
This directory contains example scripts that are useful for learning to run coffea and topcoffea. 

### Past tutorials:

* Jun 2021: [2021 TopCoffea tutoral](https://indico.cern.ch/event/1047567/)
* Aug 2022: [2022 TopCoffea tutorial Session 1](https://indico.cern.ch/event/1188768/)
* Sep 2022: [2022 TopCoffea tutorial Session 2](https://indico.cern.ch/event/1189721/)
* Jan 2023: [2023 Advanced TopCoffea tutorial](https://indico.cern.ch/event/1228170/)


### Scripts:

* `simple_processor.py` and `simple_run.py`: A minimal example of a topcoffea processor. To run:
    - Download a root file to the local directory, e.g.: `wget -nc http://www.crc.nd.edu/~kmohrman/files/root_files/for_ci/ttHJet_UL17_R1B14_NAOD-00000_10194_NDSkim.root`
    - Run the run script over a json that points to the root file you downloaded: `python simple_run.py ../../topcoffea/json/test_samples/UL17_private_ttH_for_CI.json`

