[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5258003.svg)](https://doi.org/10.5281/zenodo.5258002)
[![CI](https://github.com/TopEFT/topeft/actions/workflows/main.yml/badge.svg)](https://github.com/TopEFT/topeft/actions/workflows/main.yml)
[![Coffea-casa](https://img.shields.io/badge/launch-Coffea--casa-green)](https://cmsaf-jh.unl.edu/hub/spawn)

# topeft

`topeft` contains the Coffea processors, workflow entrypoints, histogram and
provenance contracts, plotting tools, and datacard utilities used by the
current top-quark effective-field-theory analysis. The separately
installed `topcoffea` package provides shared corrections and utilities.

## Installation orientation

For the integrated MultiCell v3 line, use the paired stack below from the
repository root on `top-26-006_multicell`. Use a fresh dedicated provider
checkout; keep legacy environments and checkouts separate:

```bash
conda env create -f environment-multicell.yml && \
conda activate coffea-multicell-env && \
TOPCOFFEA_REPO_URL=https://github.com/TopEFT/topcoffea.git \
TOPCOFFEA_GIT_REF=ef941d2cf7d93182c85ff45eb93af18fc9e6bce2 \
TOPCOFFEA_DIR="$PWD/external/topcoffea_multicell" \
  bash scripts/install_topcoffea.sh && \
python -m pip install -e .
```

The installer installs TOPCOFFEA first with the active environment's `python`;
TOPEFT is installed next in that same environment. With no explicit ref the
installer follows the upstream default branch; the fixed ref above selects the
integrated provider instead. The installer updates an existing destination,
so do not point it at a legacy checkout. See the
[migration status](MULTICELL_MIGRATION_STATUS.md) for merge identities and the
qualified environment, and [global storage options](docs/how_to/sumw2.md)
for `embedded` and `none`.

## Identify the stack for an existing v2 PKL

Start with the [legacy provenance route](docs/tutorials/histogram_artifacts.md#identify-the-stack-for-an-existing-v2-pkl).
It binds the full PKL filename and adjacent JSON sidecar to the producing
campaign attempt, both code revisions and its environment. An era, filename,
or later attempt of the same campaign cannot identify that stack. Missing or
conflicting evidence leaves it undetermined. Use a proven legacy stack for
that artifact before attempting consumption.

## Workflow orientation

The maintained high-level route begins with the `run_cr.sh` production
profile. It delegates one production block to `fullR3_run.sh`, which resolves
sample configuration and constructs a `run_analysis.py` request. The processor
produces histogram PKLs with adjacent provenance sidecars. Data-driven
transformation, plotting, datacard creation, scaling finalization, and the
later EFTFit/Combine handoff are separate downstream responsibilities.

Start with the [documentation index](docs/README.md). New analysts should use
the [analysis workflow tutorial](docs/tutorials/analysis_workflow.md); it
introduces the high-level wrapper, the lower-level wrapper, and the direct CLI
before following the artifacts downstream.

## Repository layout

- `analysis/` contains analysis entrypoints and specialist studies.
- `input_samples/` contains sample JSONs and sample-bundle configuration.
- `topeft/` is the installable Python package, including channels, parameters,
  corrections, and workflow modules.
- `tests/` contains focused regression and interface tests.
- `docs/` is the canonical reader-facing documentation system.

## Contributing and testing

Keep changes on a feature branch, add focused tests for changed behavior, and
record the validation that supports the change. See the
[testing how-to](docs/how_to/testing.md) for the maintained local commands and
the [documentation index](docs/README.md) for developer reference and
extension guidance.
