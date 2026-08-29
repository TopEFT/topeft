# Fake-tau SF fitter

The active fitter is `analysis/topeft_run2/faketau_sf_fitter.py`. It reads the
fake- and tight-tau control-region histograms from one or more pkls, computes
data and MC fake-rate points, and fits their ratio with a linear scale-factor
model. Printed tables are the primary output; `--output-json` optionally writes
the fitted TauFakeSF payload.

The direct interface owns input normalization and compatible merge, exact
histogram/axis checks, year filtering, split-or-aggregate channel resolution,
weighted fake-rate construction, the linear fit, and optional JSON
serialization. It reads channel expectations from `topeft/channels/ch_lst.json`
unless `--channels-json` selects another complete registry. It does not own tau
control-region event selection, histogram production, the sumw2 storage policy,
payload installation, or the downstream correction-consumer decision.

| Setting | Current default or authority | Consequence |
| --- | --- | --- |
| input | `histos/plotsTopEFT.pkl.gz` | a convenience only; pass `-f` explicitly for a maintained fit |
| channel configuration | `TAU_CH_LST_CR["2los_1tau"]` in `ch_lst.json` | establishes expected Ftau/Ttau split and aggregate labels |
| years | no filter | all compatible MC/data process labels in the selected artifact participate |
| regrouped tau-pT edges | `[20, 30, 40, 50, 60, 200]` in the fitter | native source bins must support the regrouping |
| uncertainty | required `tau0Fpt_sumw2` and `tau0Tpt_sumw2` | no count-based or nominal-content fallback |
| JSON output | disabled unless `--output-json` is passed | console tables remain the default result |

> **Migration:** direct `tauFitter.py` execution is deprecated and aborts
> without producing a fit or output. Use `faketau_sf_fitter.py` for maintained
> fake-tau scale-factor extraction.

The fitter resolves its configured tau control-region expectations against the
actual `channel` axes in the input histograms. A filename, directory name, or
production tag is not evidence that the required histograms or channels are
present.

## Input histogram contract

Each input pkl must contain both active histogram keys:

* `tau0Fpt`, with axes `process`, `channel`, `systematic`, and `tau0Fpt`;
* `tau0Tpt`, with axes `process`, `channel`, `systematic`, and `tau0Tpt`.

Both `tau0Fpt_sumw2` and `tau0Tpt_sumw2` companions are mandatory supported
inputs. No count-based uncertainty fallback exists, and the fitter does not
reconstruct their uncertainty from nominal yields. Produce a `taufitter`-mode
artifact with both selected companions. See
[select or change sumw2 storage](sumw2.md) for the shared companion production
contract.

For multiple input pkls, required `tau0Fpt`, `tau0Tpt`, `tau0Fpt_sumw2`, and
`tau0Tpt_sumw2` histograms are combined only after schema and merge consistency
checks. Companions must be present in every input and are combined by direct
addition, the quadrature-equivalent operation for stored sums of squared
weights. Missing or incompatible companions raise an error naming the affected
input; no supported fallback is available.

Configured Ftau/Ttau channel expectations come from
`TAU_CH_LST_CR["2los_1tau"]` in `topeft/channels/ch_lst.json`, or from the file
selected with `--channels-json`. The fitter then compares those expectations to
the channel labels actually stored in each histogram.

## Flavor-split and aggregate tau channels

The fitter supports two channel-axis layouts. A complete flavor-split layout
contains, for example:

```text
2los_ee_1tau_Ftau_2j
2los_em_1tau_Ftau_2j
2los_mm_1tau_Ftau_2j
2los_ee_1tau_Ttau_2j
2los_em_1tau_Ttau_2j
2los_mm_1tau_Ttau_2j
```

An aggregate layout may instead contain:

```text
2los_1tau_Ftau_2j
2los_1tau_Ttau_2j
```

Resolution is performed independently for each fake/tight family and jet bin:

1. If all configured flavor-split bins are present, the fitter uses them.
2. If split coverage is incomplete or absent and the corresponding aggregate
   bin is present, the fitter uses that aggregate bin.
3. If complete split coverage and the aggregate bin are both available, only
   the split bins are selected. The aggregate bin is not added, so events are
   not double-counted.
4. If neither complete split coverage nor an aggregate fallback is available,
   the fitter raises an error naming the missing split bins, aggregate bin
   checked, available channels, and corrective options.

Lepton-flavor splitting is therefore not required for fake-tau SF extraction.
There is no need to regenerate a pkl solely to add `ee/em/mm` tau channels when
the corresponding aggregate Ftau/Ttau bins are present and otherwise suitable.

For example, this input is valid even though the configured split Ftau bins are
missing:

```text
Missing configured split bins:
  2los_ee_1tau_Ftau_2j
  2los_em_1tau_Ftau_2j
  2los_mm_1tau_Ftau_2j

Available histogram bins:
  2los_1tau_Ftau_2j
  2los_1tau_Ttau_2j
  2los_CRZ_2j
```

The fitter selects `2los_1tau_Ftau_2j` for the fake family and the matching
aggregate tight bin for the tight family. Regeneration is required only when a
family has neither complete split bins nor its aggregate fallback.

## Inspect channels before fitting

`--dump-channels` prints or writes the Ftau/Ttau channel names derived from the
channel configuration. It does not inspect the pkl input(s) and does not prove
that those labels exist in the histogram axes:

```bash
python analysis/topeft_run2/faketau_sf_fitter.py \
  -f /path/to/plotsTopEFT.pkl.gz \
  --dump-channels configured_tau_channels.json
```

Inspect the actual `tau0Fpt` and `tau0Tpt` channel axes separately with the
read-only pkl inspector:

```bash
python analysis/topeft_run2/inspect_histeft_pkl.py \
  /path/to/plotsTopEFT.pkl.gz --hist tau0Fpt --max-labels 100

python analysis/topeft_run2/inspect_histeft_pkl.py \
  /path/to/plotsTopEFT.pkl.gz --hist tau0Tpt --max-labels 100
```

Before running a fit:

* confirm that `tau0Fpt` and `tau0Tpt` are top-level histogram keys;
* inspect both actual `channel` axes;
* confirm that each Ftau/Ttau family has complete split bins or its aggregate
  fallback;
* confirm `tau0Fpt_sumw2` and `tau0Tpt_sumw2` availability when weighted
  statistical uncertainties are required;
* check the selected `--year` values and the printed retained/removed process
  summary;
* when INFO logging is enabled, review the `Resolved tau CR channels` message;
  programmatic callers can inspect `stage_details["tau_channel_resolution"]` for
  the selected bins, resolution mode, and missing split-bin details;
* do not rely on filename tags as proof of histogram contents.

## Run the fitter

Run from the repository root so relative paths resolve correctly:

```bash
python analysis/topeft_run2/faketau_sf_fitter.py \
  -f /path/to/plotsTopEFT.pkl.gz \
  --channels-json /path/to/ch_lst.json
```

To combine several compatible histogram pkls before fitting, pass all paths
after one `-f`/`--pkl-file-path`:

```bash
python analysis/topeft_run2/faketau_sf_fitter.py \
  -f /path/to/plotsTopEFT_part1.pkl.gz /path/to/plotsTopEFT_part2.pkl.gz \
  --channels-json /path/to/ch_lst.json
```

The combined histograms are processed by the same split-first /
aggregate-fallback channel-resolution logic as a single input pkl. The fitter
prints a concise input summary listing the input paths, required histograms, and
whether each sumw² companion is present in all inputs or absent from all inputs.

To restrict both MC and data to selected campaign tokens:

```bash
python analysis/topeft_run2/faketau_sf_fitter.py \
  -f /path/to/plotsTopEFT.pkl.gz \
  -y 2017 2018
```

Supported values include `2016`, `2016APV`, `2017`, `2018`, `2022`, `2022EE`,
`2023`, and `2023BPix`. The fitter prints the retained and removed processes.

To write the fitted scale factors in the TauFakeSF JSON layout:

```bash
python analysis/topeft_run2/faketau_sf_fitter.py \
  -f /path/to/plotsTopEFT.pkl.gz \
  --output-json TauFakeSF_2018.json
```

The regrouped tau-pT binning defaults to `[20, 30, 40, 50, 60, 200]` and is
derived from the input histogram. Underflow and overflow are folded into the
physical range before fake rates are computed.

## Understand the output

The console output documents the major processing stages:

* **Native yield tables** list fake and tight yields and uncertainties in the
  original tau-pT bins for MC and data.
* **Regrouped fake-rate inputs** show how native bins are merged and how yields
  and uncertainties combine.
* **Fake rates by tau pT bin** show the tight/fake ratios for MC and data.
* **Scale factors (data/MC)** are the ratios used by the fit.
* **Scale-factor fit summary** lists the linear parameters, uncertainties, and
  nominal/up/down values at the representative tau-pT points.

The internal `tau_channel_resolution` stage detail records, for Ftau and Ttau,
the selected bins, `flavor_split` or `aggregate` resolution mode, missing split
bins, and aggregate fallback status.

## Modify or extend the maintained fitter

Keep configuration and calculation responsibilities distinct:

1. Change expected tau control labels in the authoritative
   `TAU_CH_LST_CR["2los_1tau"]` registry, or pass a complete reviewed registry
   through `--channels-json`; do not add a second label list to the fitter.
2. Change the regrouped pT view only at `TAU_PT_BIN_EDGES` and validate that
   every source histogram can be exactly regrouped with the intended flow
   treatment. This changes reported points and the JSON payload bins but does
   not change the stored processing histogram.
3. Add a CLI selector only for a user operation the fitter truly owns. Define
   its parser default, validate it before fitting, include it in printed or JSON
   provenance when it changes the result, and test the default and override.
4. Preserve exact nominal/sumw2 axis alignment, process/year filtering,
   split-first/aggregate-fallback selection, native-to-regrouped propagation,
   invalid-point filtering, and the requirement that at least two valid points
   remain for a linear fit.
5. Treat a change to fake-rate algebra, MC/data grouping, fit model, or
   up/down construction as a physics-policy change requiring dedicated review;
   it is not merely a wrapper extension.

Use `tests/test_faketau_sf_fitter_cli.py` for parser/output behavior,
`tests/test_faketau_sf_fitter_input_loading.py` for multi-input and companion
contracts, `tests/test_faketau_sf_fitter_channel_coverage.py` for split and
aggregate resolution, and `tests/test_fake_tau_sf_taufitter_policy.py` for the
production-policy coupling. Also inspect one representative artifact's actual
axes and retain the printed input, channel-resolution, regrouping, and fit
summaries with any payload used downstream.

A change can affect TauFakeSF JSON bin keys and values, their statistical
uncertainties, and every correction consumer that installs the payload. The
fitter writes a payload file but deliberately does not install it or certify a
campaign-wide physics result.
