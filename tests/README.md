# pytest
## Installing
Run `conda install pytest` to install
Optionally `conda install pytest-cov` to generate a local coverage report

## Running
Run `pytest` from the root directory of topcoffea. It will find the `tests` folder, and run all python scripts starting with `test_`.

## Local reports
Running `pytest --cov` will print a report at the end.
Running `pytest --cov --cov-report html` will generate an html directory that can be copied to `~/www/` to view in the browser.
By default, pytest only prints failure messages. The flags `-rP` will print all messages (`-r` is for better formatting, and `-P` is for printing messages from tests which passed).

## Contents of the `tests` folder
### `tests/test_make_1d_quad_plots.py`
Test the quadratic fit plotting script
### `tests/test_futures.py`
The `test_futures` runs the analysis processor over a given file, `test_nonprompt()` runs the output from `test_topcoffea()` through `topcoffea/modules/dataDrivenEstimation.py`, `test_datacardmaker()` creates a test datacard. 
### `test_work_queue()`
Runs the main processor with the work queue executor. 
### `test_yields()`
Checks the yields from the output pkl file from the processor against ref yields. 


Any of these test can be run individually by running e.g.
```python
python -i tests/test_topcoffea.py 
test_topcoffea()
```
 If a test requires ROOT, install with `conda install root_base -c conda-forge`

