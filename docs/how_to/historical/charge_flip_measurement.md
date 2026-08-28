# Legacy charge-flip measurement

> **Archival, unsupported interface.** These scripts remain under
> `analysis/flip_measurement` but have not been updated since the August 2023
> `topcoffea` refactoring. They document an older measurement/validation design,
> not the current TOP-26-006 charge-flip production contract.

The archived directory contains separate processors and plotters for measuring
and validating the electron charge-misidentification probability.

## Processors

* `flip_mr_processor.py` measures flip probabilities in Monte Carlo (MC)
  samples. It uses generator-level truth information to count reconstructed
  electrons whose charge is misidentified, then stores the result in a
  two-dimensional `coffea` histogram with `pt` and `abs(eta)` axes.

* `flip_ar_processor.py` validates the measurement in data. It applies the
  measured probabilities to opposite-sign (OS) events to predict same-sign
  (SS) events in the flip control region (CR).

## Plotters

* `flip_mr_plotter.py` reads the `flip_mr_processor.py` output, plots the
  two-dimensional `pt`-`abs(eta)` histograms, and serializes them as PKL files.
  Historically, operators copied those payloads to
  `topcoffea/data/fliprates` for use by `corrections.py`.

* `flip_ar_plotter.py` plots the `flip_ar_processor.py` histograms for a direct
  comparison of SS data with the charge-flip prediction. The full `topeft`
  flip-CR view was the more complete validation because it also included other
  predicted contributions, including nonprompt leptons.

## Run script

The archived `run_flip.py` selects the measurement or application processor
and its executor.

Before any attempted reuse, review the source, its sample and correction
payload assumptions, the current `topcoffea` APIs, and the current data-driven
artifact contract. Do not copy a produced payload into a maintained correction
location based on this archival page alone.
