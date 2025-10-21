# Diboson $N_{\text{jets}}$ scale factors (Run 3)

`diboson_sf_run3.py` derives diboson scale factors from the `njets` distribution.
The default binning is `[0, 1, 2, 3, 4, 5, 6]`.

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
artifacts for a year stay together.
