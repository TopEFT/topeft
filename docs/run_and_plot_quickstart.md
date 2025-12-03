# Run and plot quickstart (legacy appendix)

This page is no longer the primary starting point for new users. Follow the
[Run 2 quickstart pipeline](quickstart_run2.md) for the current end-to-end
instructions (environment → run → plot). The sections below collect a few extra
plotting tips that did not fit in the main quickstart yet; they are safe to skip
unless you need the additional context.

## Additional plotting tips

- Re-run the plotting helper with different naming/era options to keep multiple
  smoke tests separate:

  ```bash
  cd analysis/topeft_run2
  python make_cr_and_sr_plots.py \
      -f histos/local_debug/plotsTopEFT.pkl.gz \
      -o plots/local_debug \
      -n plots \
      -y 2017 \
      --skip-syst
  ```

- Inspect tuple summaries directly from Python when experimenting in notebooks:

  ```python
  import gzip, pickle
  from topeft.modules.runner_output import materialise_tuple_dict

  with gzip.open("histos/local_debug/plotsTopEFT.pkl.gz", "rb") as handle:
      tuple_summary = materialise_tuple_dict(pickle.load(handle))
  print(list(tuple_summary.items())[:2])
  ```

  The summaries make it easy to confirm that `(variable, channel, application,
  sample, systematic)` tuples match expectations before writing custom plotting
  helpers.

This file will be retired once the remaining tips are merged into the main
quickstart during the later 6B.* steps.
