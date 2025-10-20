# How to fit the results

The first step is to produce the datacard text files and root files that combine will use, and this step takes place within `topcoffea`.  The next step is to run combine, which takes place inside of a CMSSW release, outside of `topcoffea`.

## Creating the datacards

The first step is to produce the datacard text files and root files that combine will use. This step takes place within `topcoffea`. Run the `make_cards.py` script to produce the data cards.

Notes for ND users: When running steps involving condor, if you want to write to your `afs` area you will need to make sure that the permissions in your `afs` are are set properly, as outlined in the ND T3 [documentation](https://docs.crc.nd.edu/resources/NDCMS/ndcms.html#setting-up-environment). It is easier to write to somewhere besides your `afs` area, e.g. your area is `/scratch365`.

Example of running the `make_cards.py` script:
```
python make_cards.py path/to/your.pkl.gz -C --do-nuisance --var-lst lj0pt ptz -d /scratch365/yourusername/some/dir
```

## Running combine

 The next step is to run combine. This takes place inside of a CMSSW release, outside of `topcoffea`. See the [EFTFit](https://github.com/TopEFT/EFTFit) repo for instructions.

## Tau fake-rate fitter

The `analysis/topeft_run2/tauFitter.py` utility extracts fake-tau control regions
from the histogram pickle produced by `topcoffea` and fits a linear
scale-factor model.  The script operates entirely on the pickle contents—it
does not write output files—so the printed tables are the main products.

### Inputs

* A histogram pickle (`plotsTopEFT.pkl.gz` by default) containing the
  Ftau/Ttau control-region histograms with the required axes:
  `process`, `channel`, `systematic`, and `tau0pt`.
* The standard tau channel configuration JSON
  (`topeft/channels/ch_lst.json`). Use `--channels-json` to point to a custom
  configuration, and `--dump-channels OUTPUT.json` to inspect the resolved Ftau
  and Ttau channel names.

### Running the script

Run the fitter from the repository root so relative paths resolve correctly:

```bash
python analysis/topeft_run2/tauFitter.py \
  -f /path/to/plotsTopEFT.pkl.gz \
  --channels-json /path/to/ch_lst.json
```

The regrouped tau-pT binning defaults to
`[20, 30, 40, 50, 60, 80, 100, 200]` and is automatically derived from the
input histogram.  If the histogram includes under/overflow bins they are folded
into the physical range before the fake rates are computed.

### Understanding the output

The console output documents each processing step:

* **Native yield tables** list the fake and tight yields (with uncertainties)
  in the original tau-pT binning for both MC and data.
* **Regrouped fake-rate inputs** summarise how the native bins are merged into
  the working bins and display the summed yields and quadrature-combined
  uncertainties for each regrouped bin.
* **Fake rates by tau pT bin** prints the per-bin tight/fake ratios and their
  propagated uncertainties for MC and data.
* **Scale factors (data/MC)** reports the data-to-MC fake-rate ratios used in
  the fit.
* **Scale-factor fit summary** lists the fitted linear parameters (`c0` and
  `c1`), their uncertainties, and the resulting scale factors (including the
  up/down variations) evaluated at the representative tau-pT values.

These tables provide a quick sanity check for the regrouping, error propagation
and final fit behaviour without needing to inspect intermediate arrays.
