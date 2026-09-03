# MultiCell producer setup

This guide installs the MultiCell-only TopEFT producer in a permanent development directory. It uses the validated environment in `environment-multicell.yml` and pins the required `topcoffea` implementation.

## Requirements

You need:

- Git
- Conda
- Curl
- Access to GitHub, Conda-forge, and PyPI
- A Bash-compatible shell

The local validation uses the `futures` executor and does not require Condor or Work Queue.

## Install TopEFT

Choose a persistent filesystem with enough space, then clone the handoff branch:

```bash
export INSTALL_ROOT="$PWD/multicell-work"
mkdir -p "$INSTALL_ROOT"

git clone \
  --branch feature/multicell-only-producer \
  --single-branch \
  https://github.com/lazdlie/topeft.git \
  "$INSTALL_ROOT/topeft"

cd "$INSTALL_ROOT/topeft"
```

Create and activate the validated environment:

```bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1

conda env create -f environment-multicell.yml
conda activate coffea-multicell-env
```

If an environment named `coffea-multicell-env` already exists, inspect it before deciding whether to reuse it.

## Install the pinned topcoffea version

Clone `topcoffea` inside the TopEFT checkout and select the validated commit:

```bash
mkdir -p external

git clone \
  --branch run3_test_mmerged \
  --single-branch \
  https://github.com/lazdlie/topcoffea.git \
  external/topcoffea

git -C external/topcoffea checkout --detach \
  9094af8ff75b6240114041eef97d2ed36c1bfdf7

test "$(git -C external/topcoffea rev-parse HEAD)" = \
  9094af8ff75b6240114041eef97d2ed36c1bfdf7
```

Install both repositories in editable mode:

```bash
python -m pip install -e external/topcoffea
python -m pip install -e .

python -c "import topeft, topcoffea; print(topeft.__file__); print(topcoffea.__file__)"
```

`external/topcoffea` is a separate Git checkout and is intentionally ignored by the TopEFT repository.

## Run the focused tests

From the TopEFT repository root:

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_analysis_processor_hist_filter.py \
  tests/test_run_analysis_cli_help.py \
  tests/test_run_analysis_hist_outputs.py
```

Expected result:

```text
21 passed
```

These tests cover the MultiCell 1D layout, embedded nominal `sumw2`, 2D Weight storage, accumulation, serialization, histogram filtering, and removal of the Legacy backend option.

## Run a 10-event smoke test

Create a temporary directory for the test input and output:

```bash
export SMOKE_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/topeft-multicell-smoke.XXXXXXXX")"
mkdir -p "$SMOKE_ROOT/input" "$SMOKE_ROOT/output"

export INPUT="$SMOKE_ROOT/input/ttHJet_UL17_R1B14_NAOD-00000_10194_NDSkim.root"

curl --fail --location \
  --output "$INPUT" \
  http://www.crc.nd.edu/~kmohrman/files/root_files/for_ci/ttHJet_UL17_R1B14_NAOD-00000_10194_NDSkim.root
```

Optionally verify the input checksum:

```bash
test "$(python -c \
  'import hashlib, sys; print(hashlib.file_digest(open(sys.argv[1], "rb"), "sha256").hexdigest())' \
  "$INPUT")" = \
  850448f4844001595e4276cbab9fcb926f30458f92d8f75c251a1823b54a8624
```

Run one 10-event chunk:

```bash
PYTHONPATH="$PWD" python analysis/topeft_run2/run_analysis.py \
  input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json \
  --executor futures \
  --prefix "$SMOKE_ROOT/input/" \
  --nworkers 1 \
  --nchunks 1 \
  --chunksize 10 \
  --hist-list njets \
  --outpath "$SMOKE_ROOT/output" \
  --outname njets-smoke
```

Validate the output:

```bash
PYTHONPATH="$PWD" python - <<'PY'
import gzip
import os

import cloudpickle
import numpy as np

from topcoffea.modules.histEFT import HistEFT

path = os.path.join(
    os.environ["SMOKE_ROOT"],
    "output",
    "njets-smoke.pkl.gz",
)

with gzip.open(path, "rb") as stream:
    output = cloudpickle.load(stream)

assert set(output) == {"njets"}
assert not any(name.endswith("_sumw2") for name in output)

njets = output["njets"]
assert isinstance(njets, HistEFT)
assert njets._use_multicell is True
assert njets.store_sumw2 is True

sumw2 = sum(
    np.asarray(values).sum()
    for values in njets.nominal_sumw2().values()
)
assert np.isfinite(sumw2) and sumw2 > 0

print("MultiCell smoke test passed")
PY
```

## Work Queue note

The local installation and `futures` producer workflow are reproducible with this setup.

Work Queue execution additionally depends on site-specific Condor configuration, worker provisioning, authentication, networking, storage, and remote ROOT access. The Conda-forge XRootD builds tested during this handoff segfaulted on Glados, so remote XRootD I/O is not currently claimed as reproducible.

Before using `--executor work_queue`, validate the site-supported XRootD installation and worker configuration.
