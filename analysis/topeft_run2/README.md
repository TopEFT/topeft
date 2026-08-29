# Analysis entry-point directory

This directory contains current and historical analysis executables. It is a
source-location map, not a second user manual. The canonical documentation
starts at [`docs/README.md`](../../docs/README.md).

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
`run_plotter.sh`/`make_cr_and_sr_plots.py`, `make_cards.py`, and
`datacards_post_processing.py` own distinct downstream stages; they are not
alternate processor entry points.

`fullR2_run.sh` and the `--set-up-top22006` card topology are retained for
historical TOP-22-006 reproduction. Their support boundary is documented
separately in the
[historical TOP-22-006 guide](../../docs/how_to/historical/top_22_006.md).
