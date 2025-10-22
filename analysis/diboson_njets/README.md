# Diboson $N_{\text{jets}}$ scale factors (Run 3)

`diboson_sf_run3.py` derives diboson scale factors from the `njets` distribution.
The default binning is `[0, 1, 2, 3, 4, 5, 6]`.

## Scale factor calculation

For each requested year, the script loads the `njets` histogram, removes any
flip-control samples (`flip*` processes), and subtracts the remaining
non-diboson backgrounds from the data template.  The diboson prediction is
built by summing the `WZTo*`, `ZZTo*`, and `WWTo*` processes, so the scale
factors are computed as `(data − other) / diboson` on a bin-by-bin basis.
Both the filtering of the diboson samples and the flip-sample exclusion are
implemented in `process_year` within `diboson_sf_run3.py`.

## Shared pickle workflow

Run 3 histogram production commonly yields one pickle containing every year.
To run the script once and obtain scale factors for all encoded years:

1. Ensure the combined pickle embeds the year information in the process names
   (tokens such as `central2023` or `2022EE` work well).
2. Execute the script with `--pkl` pointing to the shared file and `--year all`.
   The script discovers every year token automatically, fits them individually,
   and also produces a combined summary across the processed years.
3. Inspect the per-year directories created beneath `--output-dir`.  Each
   directory contains `diboson_sf_{year}.json`, the linear-fit JSON, and the
   diagnostic PNG, so every year's artifacts stay grouped together.

## Input options

* **Per-year pickles** – supply one histogram pickle per year:
  ```bash
  python diboson_sf_run3.py --pkl 2022.pkl.gz 2023.pkl.gz --year 2022 2023
  ```
* **Templated paths** – provide a path template with `{year}` that expands for
  each requested year:
  ```bash
  python diboson_sf_run3.py --pkl "/path/to/year_{year}.pkl.gz" --year 2022 2023
  ```
* **Shared pickle** – give a single file that contains histograms for multiple
  years.  The script matches process names that encode each year (tokens such as
  `central2023` or `2022EE`).  If that strict pattern fails, a substring search
  is used as a fallback so unexpected naming conventions still match:
  ```bash
  python diboson_sf_run3.py --pkl combined.pkl.gz --year 2022 2023
  ```
* **Automatic discovery** – use `--year all` to scan the shared pickle for every
  year token.  The script runs each detected year individually and also writes a
  combined result across all processed years:
  ```bash
  python diboson_sf_run3.py --pkl combined.pkl.gz --year all
  ```

## Outputs

Per-year results (and the combined entry when `--year all` is used) are written
under subdirectories of `--output-dir`.  Each directory contains
`diboson_sf_{year}.json`, the linear-fit JSON, and the PNG diagnostic plot so all
artifacts for a year stay together.  The scale-factor JSON stores the
bin-by-bin values only, while the linear-fit JSON (`diboson_sf_{year}_linear_fit.json`)
records the slope and intercept that appear in the CLI summary table.  The PNG
shows the fitted line overlaid on the scale factors, but the numeric
coefficients live solely in the linear-fit JSON (and the CLI summary’s Slope
and Intercept columns) to contextualise the reported mean scale factor.
