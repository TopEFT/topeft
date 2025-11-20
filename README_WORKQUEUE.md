# Distributed execution with TaskVine

TaskVine is the supported distributed backend for `topeft`.  The helpers in
`analysis/topeft_run2/workflow.py` and the command line interfaces default to the
TaskVine executor so that large campaigns can share workers across sites.  This
note collects the minimum steps required to stage the Python environment, launch
the manager, and submit workers.  For a more complete walkthrough see
[`docs/taskvine_workflow.md`](docs/taskvine_workflow.md) together with the main
[README](README.md).

## Environment packaging

TaskVine managers ship the same tarball built by
`topcoffea.modules.remote_environment`.  From the repository root run:

```sh
conda env create -f environment.yml           # or: conda env update -f environment.yml --prune
conda activate coffea2025
pip install -e .                              # install topeft in editable mode
pip install -e topcoffea/                     # install topcoffea in editable mode
python -m topcoffea.modules.remote_environment
```

The final command prints the path to the packaged environment.  Re-run it
whenever you update dependencies or pull new commits so the archive remains in
sync with your editable checkouts.

## Running the analysis with TaskVine

From `analysis/topeft_run2` prepare a run configuration (JSON/YAML/CFG as shown
in the quickstart docs) and launch the manager:

```sh
conda activate coffea2025
python run_analysis.py --executor taskvine --chunksize 160000 \
    ../../input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json
```

The manager listens on the port range configured via `--port` (default
`9123-9130`) and writes staging artefacts under
`${TOPEFT_EXECUTOR_STAGING:-$TMPDIR/topeft/<user>-taskvine-coffea}`.  Logs live
in the `logs/taskvine/` subdirectory within the staging area.

## Launching workers

Use the environment archive from the packaging step to start workers with
`vine_submit_workers` (TaskVine automatically distributes it if the flag is
omitted, but pre-loading the tarball avoids repeated transfers):

```sh
python_env=$(python -m topcoffea.modules.remote_environment)
vine_submit_workers --cores 4 --memory 16000 --disk 16000 \
    --python-env "$python_env" -M ${USER}-taskvine-coffea 10
```

Adjust the resources to match your site policies.  The same command works for
local testing by setting `--cores 1 --memory 4000 --disk 4000` and requesting a
single worker.

## Legacy Work Queue

Work Queue support has been removed from the workflow helpers and CLIs.  Older
instructions remain available in the Git history prior to this change.  If you
are required to operate against a Coffea build that still exposes
`WorkQueueExecutor`, pin the repository to a revision before the TaskVine-only
switch and follow the historic instructions that accompanied that release.
