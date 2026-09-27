# Analysis entry-point directory

This directory contains current and historical analysis executables. For usage,
configuration, and output details, start at
[`docs/README.md`](../../docs/README.md).

For current TOP-26-006 work:

- follow the [new-analyst tutorial](../../docs/tutorials/analysis_workflow.md);
- choose and extend production entry points with the
  [production how-to](../../docs/how_to/production.md);
- run transformed nonprompt products with the
  [nonprompt how-to](../../docs/how_to/nonprompt.md);
- use the [plotting](../../docs/how_to/plotting.md),
  [cards/scalings](../../docs/how_to/datacards_and_scalings.md),
  [sumw2](../../docs/how_to/sumw2.md), and
  [binning](../../docs/how_to/flexible_binning.md) guides for downstream tasks;
- look up exact entry-point contracts in the
  [software reference](../../docs/reference/README.md);
- read the [architecture explanation](../../docs/explanation/architecture.md)
  before changing responsibility boundaries.

The maintained high-to-low production path is `run_cr.sh` -> `fullR3_run.sh`
-> `run_analysis.py` -> `AnalysisProcessor`. `run_data_driven.py`,
`run_plotter.sh`/`make_cr_and_sr_plots.py`, `make_cards.py`,
`make_datacard_matrix_manifest.py`, the resumable datacard matrix runner,
`build_per_era_datacard_package.py`, and
`build_combined_datacard_package.py` perform distinct downstream tasks; they
are not alternate processor entry points. The builders check and package
completed producer rows by era, then assemble and check the cards-only combined
package. For direct card work, `make_cards.py` takes PKLs and writes cards,
templates, selected WCs, and preselection scalings without a manifest. For
resumable multi-row production, the manifest helper binds the Run 2 or Run 3
profile's five input roles to current PKLs and writes a
`topeft_datacard_matrix_v3` manifest. Each full profile has 11 logical rows;
the maintained partition has 11 Run 2 execution units or 34 Run 3 units.
The runner executes those units and writes receipts. Historical v2 manifests
are not accepted by the current workflow; generate a new v3 manifest rather
than converting one. The per-era builder derives its requested physical targets
from those manifest rows and reports missing or extra completions before
publishing.
The combined builder requires matching physical target sets across eras;
matching restricted subsets are supported. Its `ordered_card_inputs.txt`
lists cards in combination order.

`fullR2_run.sh` and the `--set-up-top22006` card topology are retained for
historical TOP-22-006 reproduction. Their support boundary is documented
separately in the
[historical TOP-22-006 guide](../../docs/how_to/historical/top_22_006.md).
