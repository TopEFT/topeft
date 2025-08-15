:warning: The scripts in this directory have not been updated since the `topcoffea` refactoring (of August 2023), if they are needed in the future, likely some import statements will need to be updated.

## mc_validation

This directory contains scripts from the validation studies of the FullR2 private UL MC generated for TOP-22-006 against central UL MC performed during the June 2022 MC validation studies (for TOP-22-006 pre approval checks).

* `mc_validation_gen_processor.py`:
    - This script produces gen level histograms for comparison of private and central MC

* `mc_validation_gen_plotter.py`:
    - This script makes plots to compare private and central GEN level distributions
    - Should be run on the output of the mc_validation_gen_processor.py processor

* `mc_validation_plotter.py`:
    - This script makes plots to compare private and central RECO level distributions
    - Should be run on the output of the topeft processor
    - Was used during the June 2022 MC validation studies (for TOP-22-006 pre approval checks)

:note: The scripts below were updated in August of 2025
* `gen_processor.py`:
    - This is an updated script to produce gen level histograms for comparison of various samples

* `gen_hist_eventweights_processor.py`:
    - This script produces distributions of MG weights (if they are saved in the samples)

* `gen_hist_eventweights_plotter.py`:
    - This script produces plots of MG weights
    - Should be run on the output of the gen_hist_eventweights_processor.py processor

* `comp_norm.py`:
    - This script plots two pkl files for different variables for comparison of shapes and normalizations
    - Should be run on the output of the gen_processor.py processor
