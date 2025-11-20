# Run and plot quickstart

This guide walks through creating the shared environment, running `run_analysis.py` with the local futures and TaskVine executors, and turning the resulting histogram pickles into plots with the existing helpers. All runs rely on the 5-tuple histogram schema `(variable, channel, application, sample, systematic)` documented in [analysis_processing.md](analysis_processing.md) and enforced across the plotting stack, so ensure any downstream consumers preserve those keys.

## 1) Create the environment and install dependencies

1. Install Miniconda if it is not already available:

   ```bash
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > conda-install.sh
   bash conda-install.sh
   ```

2. Create and activate the shared `coffea2025` environment, then install `topeft` in editable mode:

   ```bash
   cd /path/to/topeft
   unset PYTHONPATH
   conda env create -f environment.yml
   conda activate coffea2025
   pip install -e .
   ```

3. Clone and install the sibling [`topcoffea`](https://github.com/TopEFT/topcoffea) checkout **on the `ch_update_calcoffea` branch** so the processors and packaged environments agree on the dependency baseline:

   ```bash
   cd /path/to
   git clone https://github.com/TopEFT/topcoffea.git
   cd topcoffea
   git switch ch_update_calcoffea
   pip install -e .
   cd ../topeft
   python -c "import topcoffea"  # smoke test
   ```

4. (TaskVine only) Build a packaged environment tarball to hand to remote workers. The helper detects editable changes and rebuilds automatically:

   ```bash
   python -m topcoffea.modules.remote_environment
   ```

   The printed path (under `topeft-envs/`) is passed to TaskVine via the executor `environment_file` argument; local futures runs do **not** need this tarball because they rely on the already-activated Conda environment.

## 2) Run `run_analysis.py`

Example commands assume `analysis/topeft_run2` as the working directory and the sample manifest from `input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json`.

### Local futures executor (single node)

Use the default configuration bundle and force the futures backend for quick smoke tests:

```bash
cd analysis/topeft_run2
python run_analysis.py \
    ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
    --options configs/fullR2_run.yml:sr \
    --executor futures \
    --outpath histos/local_futures_quickstart \
    --outname plotsTopEFT
```

This path processes the signal-region profile locally and writes tuple-keyed histogram pickles such as `histos/local_futures_quickstart/plotsTopEFT.pkl.gz`.

### TaskVine executor (distributed)

Re-use the same YAML profile with the TaskVine backend enabled. Because the executor hands off work to a TaskVine manager, ensure workers connect to the advertised manager name and have access to the packaged environment tarball printed earlier.

```bash
cd analysis/topeft_run2
python run_analysis.py \
    ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
    --options configs/fullR2_run.yml:sr \
    --executor taskvine \
    --environment-file "$(python -m topcoffea.modules.remote_environment)" \
    --outpath histos/taskvine_quickstart \
    --outname plotsTopEFT
```

Start at least one worker in another terminal, matching the manager string reported by the run (for example `vine_worker --cores 1 --memory 8000 --disk 8000 -M ${USER}-taskvine-coffea`). The TaskVine invocation requires the tarball supplied through `--environment-file`; omit it only when debugging locally with the futures executor.

Both executors emit histogram pickles keyed by the `(variable, channel, application, sample, systematic)` tuples referenced above. Downstream plotting helpers rely on those keys to locate channel and systematic metadata.

## 3) Turn the histogram pickle into plots

The `analysis/topeft_run2/make_cr_and_sr_plots.py` helper consumes the tuple-keyed output and produces control- and signal-region plots. Point it at the pickle produced in the previous step:

```bash
cd analysis/topeft_run2
python make_cr_and_sr_plots.py \
    -f histos/taskvine_quickstart/plotsTopEFT.pkl.gz \
    -o plots/taskvine_quickstart \
    -n plots \
    -y 2018 \
    --skip-syst
```

For quick inspection inside a notebook or ad-hoc script, you can also materialise the tuple summary directly:

```python
import gzip
import pickle
from topeft.modules.runner_output import materialise_tuple_dict

with gzip.open("histos/local_futures_quickstart/plotsTopEFT.pkl.gz", "rb") as handle:
    histos = pickle.load(handle)

tuple_summary = materialise_tuple_dict(histos)
print(list(tuple_summary.items())[:2])
```

Both paths assume the stored histogram tuples follow the `(variable, channel, application, sample, systematic)` convention so channel- and systematic-aware plotting works as expected.
