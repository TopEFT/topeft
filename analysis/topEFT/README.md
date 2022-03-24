## topEFT
This directory contains scripts for the Full Run 2 EFT analysis. This README documents and explains how to run each of the scrips that are run from the command line.

### Plotting Scripts

* `make_cr_plots.py`:
    - This script makes plots for all CRs categories. 
    - The script takes as input a pkl file that should have both data and background MC included.
    - Example usage: `python make_cr_plots.py -f histos/your.pkl.gz -o ~/www/some/dir -n some_dir_name -y 2018 -t -u`

* `genPlot.py`:

* `plot.py`:
    - Make sure all the names in `histAxisName` in the json files are listed in `plot.py` under processDic and/or bkglist as needed
      - Example to get `histAxisName` from private TOP-19-001 json files: `grep topcoffea/json/signal_samples/private_top19001_local/*Jet*.json -e histAxisName | sed 's/^.*: "//g' | sed 's/",//g'`
    - Example usage: `python analysis/topEFT/plot.py -p histos/ttH_private_UL17_cuts_levels.pkl.gz -y 2017`

### Wrappers for processors

* `Genrun.py`:

* `run.py`:
    - This is the run script for the main `topeft.py` processor. Its usage is documented on the repository's main README. It uses the `futures` executor, with 8 cores by default. You can configure the run with a number of command line arguments, but the most important one is the config file, where you list the samples you would like to process (by pointing to the JSON files for each sample, located inside of `topcoffea/json`. 
    - Example usage: `python run.py ../../topcoffea/cfg/your_cfg.cfg`  

* `work_queue_run.py`:
    - This run script also runs the main `topeft.py` processor, but it uses the `work_queue` executor. Pass the config file to this script in exactly the same was as with `run.py`. The `work_queue` executor makes use of remote resources, and you will need to submit workers using a `condor_submit_workers` command as explained on the main `topcoffea` README.
    - Example usage: `python work_queue_run.py ../../topcoffea/cfg/your_cfg.cfg`

### Scripts for finding and comparing yields

* `get_yield_json.py`:
    - This script takes a pkl file produced by the processor, finds the yields in the analysis categories, and saves the yields to a json file. It can also print the info to the screen. The default pkl file to process is `hists/plotsTopEFT.pkl.gz`.
    - Example usage: `python get_yield_json.py -f histos/your_pkl_file.pkl.gz`

* `comp_yields.py`:
    - This script takes two json files of yields (produced by `get_yield_json.py`), finds the difference and percent difference between them in each category, and prints out all of the information. You can also compare to the TOP-19-001 yields by specifying `TOP-19-001` as one of the inputs. Specifying the second file is optional, and it will default to the reference yield file. The script returns a non-zero exit code if any of the percent differences are larger than a given value (currently set to 1e-8). 
    - Example usage: `python comp_yields.py your_yields_1.json your_yields_2.json`

* `check_yields.sh`:
    - This script calls `run.py` (to run the `topeft` processor over a sample), `get_yield_json.py` (to find the yields, and save them to a json file), and `comp_yields.py` (to compare these yields to reference yields).
    - Example usage: `source check_yields.sh`

* `remake_ci_ref_datacard.py`:
    - This script runs the datacard maker tests.
    - Example usage: `python remake_ci_ref_datacard.py`
* `remake_ci_ref_datacard.sh`:
    - This script runs `remake_ci_ref_datacard.py` and copies the resulting reference files to the `analysis/topEFT/test`
    - Example usage: `sh remake_ci_ref_datacard.sh`

### Other scripts

* `datacard_maker.py`:
    - This script produces datacards to be used in combine. It creates temporary ROOT files with `tmp` in the name, and delets them when finished. The final txt and root files are in the `histos` folder. The txt files are the datacards, and the root files are the shape files.
    - Example usage: `python analysis/topEFT/datacard_maker.py histos/all_private_njet_opt.pkl.gz`
    - More cards can be made by adding addtional calls to `analyzeChannel()` (see file for details).
    - (Optional) Specify a subset of WCs using e.g. `--POI cpt,ctp,cptb,cQlMi,cQl3i,ctlTi,ctli,cbW,cpQM,cpQ3,ctei,cQei,ctW,ctlSi,ctZ,ctG`
    - (Optional) Specify a list of variables to make cards for e.g. `--var-lst njets ptbl`
    - To run over the systematics in a pkl file, use the `--do-nuisance` flag. This will increase the processing time, but shouldn't take more than 10 miutes for all the condor jobs to finish.
    - Example outputs:
        - `ttx_multileptons-2lss_p_2b.txt` - datacard for 2lss p 2 b-jet `njets` (stores normaliztion numbers)
        -  `ttx_multileptons-2lss_p_2b.root` - shape file for 2lss p 2 b-jet `njets` (stores actual TH1F histograms)
        -  `ttx_multileptons-3l_onZ_1b_4j_ptbl.txt` - datacard for 3l on-shell Z 1 b-jet 4 jet `ptbl`

* `drawSliders.py`:
 
* `make_jsons.py`:
    - The purpose of this script is to function as a wrapper for the `topcoffea/modules/createJSON.py` script. That script can also be run from the command line, but many options must be specified, so if you would like to make multiple JSON files or if you will need to remake the JSON files at some point, it is easier to use this script.
    - To make JSON files for samples that you would like to process:
        * Make a dictionary, where the key is the name of the JSON file you would like to produce, and the value is another dictionary, specifying the path to the sample, the name you would like the sample to have on the `sample` axis of the coffea histogram, and the cross section that the sample should correspond to in the `topcoffea/cfg/xsec.cfg` file. The path to the sample should start with `/store`. The existing dictionaries in the file can be used as examples.
        * In the `main()` function, call `make_jsons_for_dict_of_samples()`, and pass your dictionary, along with a redirector (if you are accessing the sample via xrootd), the year of the sample, and the path to an output directory.
        * After the JSON file is produced, it will be moved to the output directory that you specified.
    - Make sure to run `run_sow.py` and `update_json_sow.py` to update the sum of weights before committing and pushing any updates to the json files
    - Once you have produced the JSON file, you should consider committing it to the repository (so that other people can easily process the sample as well), along with the updated `make_jsons.py` script (so that if you have to reproduce the JSON in the future, you will not have to redo any work).
    - Example usage: `python make_jsons.py`

* `run_sow.py`:
    - This script runs over the provided json files and calculates the properer sum of weights
    - Example usage: `python run_sow.py ../../topcoffea/json/signal_samples/private_UL/UL17_tHq_b1.json --xrd root://deepthought.crc.nd.edu/`

* `update_json_sow.py`:
    - This script updates the actual json files corresponding to the samples run with `run_sow.py`
    - Example usage: `python update_json_sow.py histos/sowTopEFT.pkl.gz`

* `missing_parton.py`:
    - This script compares two sets of datacards (central NLO and private LO) and computes the necessary uncertainty to bring them into agreement (after account for all included systematics).
    - Datacards should be copied to `histos/central_sm` and `histos/private_sm` respectively.
    - Example usage: `python analysis/topEFT/missing_parton.py --output-path ~/www/coffea/master/1D/ --years 2017`
