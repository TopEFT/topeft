# Run and plot quickstart (legacy appendix)

> **Legacy appendix:** kept for historical plotting tips only. Follow the
> [Run 2 quickstart pipeline](quickstart_run2.md) for the supported environment,
> run, and plot workflow. Use this page only when you need the supplementary
> plotting tricks recorded below.

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

This file is preserved for reference; the canonical Run‑2 quickstart lives in
`docs/quickstart_run2.md`.
