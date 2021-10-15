# pytest
## Installing
Run `conda install pytest` to install
Optionally `conda install pytest-cov` to generate a local coverage report

## Running
Run `pytest` from the root directory of topcoffea. This will find the `tests` folder, and run all python scripts starting with `test_`.

## Local reports
Running `pytest --cov` will print a report at the end.
Running `pytest --cov --cov-report html` will generate an html directory that can be copied to `~/www/` to view in the browser.
By default, pytest on prints failure messages. The flags `-rP` will print all messages (`-r` is for better formatting, and `-P` is for printing messages from tests which passed).

## Contents of the `tests` folder
### `tests/test_HistEFT_add.py`
 Various HistEFT tests
### `tests/test_unit.py`
Unit tests for HistEFT (ported from the C++/ROOT version of TH1ET)
### `tests/test_make_1d_quad_plots.py`
Test the quadratic fit plotting script
### `tests/test_topcoffea.py`
`test_topcoffea()` runs topcoffea over `NAOD-00000_18449.root` (run `wget http://www.crc.nd.edu/~ywan2/root_files/NAOD-00000_18449.root` to download this file.)
`test_nonprompt()` runs the output from `test_topcoffea()` through `topcoffea/modules/dataDrivenEstimation.py`
`test_make_yields()` runs the output from `test_topcoffea()` through `analysis/topEFT/get_yield_json.py` to produce the json file for comparison
`test_compare_yields()`  runs the output from `test_make_yields()` through `analysis/topEFT/comp_yields.py` to check the output yields match the reference file
`test_datacard()` runs the output from `test_topcoffea()` through the datacard maker for a few test cases, and compares the resulting datacards to the reference files
 - This test requires ROOT, install with `conda install root_base -c conda-forge`

Any of these test can be run individually by running e.g.
```python
python -i tests/test_topcoffea.py 
test_topcoffea()
```
