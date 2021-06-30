## topEFT
This directory contains scripts for the Full Run 2 EFT analysis. This README documents and explains how to run each of the scrips that are run from the command line.

### Plotting Scripts

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

### Other scripts

* `convert3lEFT.py`:
    - [Depreciated] Will remove once the Datacard Maker is ready

* `drawSliders.py`:

* `find_yields.py`:
    - This script is a work in progress. The main functionality of this script is to load the histograms from the pkl files produced by the processor, and to find the yield for each category that is specified in the `CATEGORIES` dictionary. Currently, these categories are designed to match the categories defined in TOP-19-001. Once you have obtained the yields (via the `get_yld_dict()` function), you can manipulate them (e.g. find the percent different between two yield dictionaries using `get_pdiff_between_nested_dicts()`) or display them (by dumping them to the screen with `print_yld_dicts()`, or dumping them to the screen in the format of a latex table with `print_latex_yield_table()`).
    - Example usage: `python find_yields.py`
 
* `make_jsons.py`:
    - The purpose of this script is to function as a wrapper for the `topcoffea/modules/createJSON.py` script. That script can also be run from the command line, but many options must be specified, so if you would like to make multiple JSON files or if you will need to remake the JSON files at some point, it is easier to use this script.
    - To make JSON files for samples that you would like to process:
        * Make a dictionary, where the key is the name of the JSON file you would like to produce, and the value is another dictionary, specifying the path to the sample, the name you would like the sample to have on the `sample` axis of the coffea histogram, and the cross section that the sample should correspond to in the `topcoffea/cfg/xsec.cfg` file. The path to the sample should start with `/store`. The existing dictionaries in the file can be used as examples.
        * In the `main()` function, call `make_jsons_for_dict_of_samples()`, and pass your dictionary, along with a redirector (if you are accessing the sample via xrootd), the year of the sample, and the path to an output directory.
        * After the JSON file is produced, it will be moved to the output directory that you specified.
    - Once you have produced the JSON file, you should consider committing it to the repository (so that other people can easily process the sample as well), along with the updated `make_jsons.py` script (so that if you have to reproduce the JSON in the future, you will not have to redo any work).
    - Example usage: `python make_jsons.py`
