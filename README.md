[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5258003.svg)](https://doi.org/10.5281/zenodo.5258002)
[![CI](https://github.com/TopEFT/topeft/actions/workflows/main.yml/badge.svg)](https://github.com/TopEFT/topeft/actions/workflows/main.yml)
[![Coffea-casa](https://img.shields.io/badge/launch-Coffea--casa-green)](https://cmsaf-jh.unl.edu/hub/spawn)

# topeft

`topeft` contains the Coffea processors, workflow entrypoints, histogram and
provenance contracts, plotting tools, and datacard utilities used by the
current TOP-26-006 top-quark effective-field-theory analysis. The separately
installed `topcoffea` package provides shared corrections and utilities.

## Installation orientation

A local development environment starts from the tracked environment and an
editable `topeft` installation:

```bash
conda env create -f environment.yml
conda activate coffea-env
pip install -e .
scripts/install_topcoffea.sh
```

The installer prepares the matching editable `topcoffea` checkout and its
required data payloads. Campaign operators should record the exact repository
revisions and environment archive used for production.

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
