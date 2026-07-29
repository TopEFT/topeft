# HistEFT and pkl inspection tutorial

## 1. Purpose and scope

This note is an onboarding guide for the current TOP EFT histogram workflow:

```text
analysis processor -> HistEFT/SparseHist output -> pkl.gz file -> plotting or manual inspection
```

It is intentionally source-grounded. File and line references point to the
implementation that was inspected for this guide. The goal is to make a new
student autonomous enough to inspect pkl files and understand the current
contracts before planning a future `scikit-hist` EFT-aware replacement.

This guide does not change any physics behavior. It does not redesign HistEFT,
the processor, `run_cr.sh`, `run_analysis.py`, plotting code, sample JSONs, or
CFG files. The only runner ergonomics change documented here is an opt-in
`fullR3_run.sh` input override for tutorial-scale dry-runs and single-sample
tests; production defaults are unchanged.

## Companion API contract

For the formal compatibility contract, implemented-vs-used feature matrix, EFT
semantics, pkl compatibility requirements, and parity-test specification, see
[`docs/histeft_api_contract.md`](histeft_api_contract.md).

Read this tutorial first if you are learning how to run the processor and
inspect pkls. Read the API contract next if you are planning a future replacement
of HistEFT or writing parity tests.

## 2. Big picture: processor -> HistEFT/coffea output -> pkl -> plotting/inspection

The current workflow has four layers:

1. `analysis/topeft_run2/run_cr.sh` is the student-facing source of truth for
   how this workspace is meant to launch the processor. It delegates to
   `analysis/topeft_run2/fullR3_run.sh`, which builds a `run_analysis.py`
   command.
2. `analysis/topeft_run2/run_analysis.py` reads JSON or CFG sample inputs,
   builds a sample dictionary and WC list, instantiates
   `AnalysisProcessor`, runs coffea, and writes a gzip-compressed pkl with
   `cloudpickle`.
3. `analysis/topeft_run2/analysis_processor.py` declares one histogram per
   requested variable. One-dimensional analysis variables are `HistEFT`
   objects. Two-dimensional variables are `SparseHist` objects.
4. `analysis/topeft_run2/make_cr_and_sr_plots.py` loads one or more pkl files,
   validates and merges histograms, groups process labels, integrates channels
   and systematics, evaluates HistEFT at SM or WC points, and produces plots.

Manual inspection only needs the pkl file and the histogram object API. The
helper added with this guide,
`analysis/topeft_run2/inspect_histeft_pkl.py`, gives a safe first look without
depending on the plotting script.

## 3. What HistEFT is

`HistEFT` is a histogram class for storing EFT polynomial coefficients instead
of only one nominal bin content per bin.

In a normal weighted histogram, each event contributes one weight to one dense
bin. In `HistEFT`, each event contributes one value for every quadratic EFT
coefficient term. With `n` Wilson coefficients, the number of stored quadratic
terms is the lower-triangular count for `sm` plus all WCs. For example, with
one WC the terms are `sm*sm`, `ctG*sm`, and `ctG*ctG`.

Important source behavior:

- `topcoffea/topcoffea/modules/histEFT.py:74-126`: `HistEFT` requires named
  axes, exactly one user dense axis, that dense axis last, categorical axes with
  growth, and `Double` storage. It creates or accepts an internal
  `quadratic_term` axis.
- `topcoffea/topcoffea/modules/histEFT.py:140-163`: maps WC pairs such as
  `("sm", "ctG")` or `("ctG", "ctG")` to a quadratic-term index.
- `topcoffea/topcoffea/modules/histEFT.py:197-249`: `fill` repeats dense
  values and event weights across all quadratic terms, then fills the internal
  `quadratic_term` axis with EFT coefficients multiplied by the event weight.
- `topcoffea/topcoffea/modules/histEFT.py:271-305`: `eval({})` evaluates the
  stored polynomial at the SM point, while `eval({"ctG": 1.0})` evaluates at a
  specific WC point. `as_hist(values)` materializes a regular histogram after
  evaluation.

The practical consequence: for a `HistEFT` object, raw stored values are not yet
the final physics yield at an arbitrary EFT point. Plotting and manual
inspection must either evaluate it at a WC point or explicitly inspect the raw
coefficient axis.

## 4. Where HistEFT lives in the code

The implementation is in the sibling `topcoffea` repository:

```text
/users/apiccine/work/correction-lib/topcoffea/topcoffea/modules/histEFT.py
/users/apiccine/work/correction-lib/topcoffea/topcoffea/modules/sparseHist.py
```

`HistEFT` inherits from `SparseHist`. `SparseHist` is the sparse categorical
storage layer: it tracks categorical axes in a small bookkeeping histogram and
stores one dense `hist.Hist` block per populated categorical key. The relevant
source is:

- `topcoffea/topcoffea/modules/sparseHist.py:15-39`: class setup, categorical
  axes, dense axes, and `_dense_hists`.
- `topcoffea/topcoffea/modules/sparseHist.py:124-139`: fill bookkeeping and
  per-key dense histogram creation.
- `topcoffea/topcoffea/modules/sparseHist.py:299-325`: slicing/integration
  return either a dense histogram, a new sparse histogram, or a scalar.
- `topcoffea/topcoffea/modules/sparseHist.py:349-376`: `values`, `view`, and
  `integrate`.
- `topcoffea/topcoffea/modules/sparseHist.py:378-406`: grouping categorical
  bins.
- `topcoffea/topcoffea/modules/sparseHist.py:445-529`: arithmetic and pickle
  reconstruction.

EFT coefficient helper logic is split across:

- `topcoffea/topcoffea/modules/quad_fit_tools.py:203-240`: extracts
  `EFTfitCoefficients` from events and defines the quadratic coefficient order.
- `topcoffea/topcoffea/modules/eft_helper.py:208-266`: remaps coefficient
  arrays when a histogram WC list differs from the sample WC list.

## 5. HistEFT data model

### Axes

The processor constructs a `HistEFT` with four sparse categorical axes and one
analysis dense axis:

```text
process, channel, systematic, appl, <analysis variable>
```

Source:

- `analysis/topeft_run2/analysis_processor.py:212-236`: declares the
  categorical axes `process`, `channel`, `systematic`, and `appl`, then builds
  histogram-variable names from `topeft/modules/axes.py`.
- `analysis/topeft_run2/analysis_processor.py:393-431`: creates scalar and/or
  EFT 1D siblings and policy-selected scalar `<name>_sumw2` companions.
- `topeft/modules/axes.py:1-32`: examples of one-dimensional variable
  definitions such as `invmass`, `ptz`, and `njets`.
- `topeft/modules/axes.py:230-260`: two-dimensional variable definitions use a
  separate `info_2d` dictionary.

`HistEFT` also carries an internal dense `quadratic_term` axis. That axis is
created in `histEFT.py:105-126` and is included in the dense storage layer, but
the user-facing analysis dense axis is still the physics variable such as
`njets` or `ptz`.

### Dense vs sparse axes

`SparseHist` treats categorical axes as sparse and dense axes as regular
histogram axes. In this codebase:

- sparse axes: process, channel, systematic, appl;
- dense physics axis: one requested analysis variable;
- dense internal EFT axis: `quadratic_term`, managed by `HistEFT`.

Only populated sparse category combinations get dense histogram blocks. That is
why manual pkl inspection should not assume every process/channel/systematic
combination exists.

### Sample and process labels

The processor uses sample JSON metadata to assign the process axis label:

- `analysis/topeft_run2/analysis_processor.py:450-466`: reads `dataset`,
  `histAxisName`, `year`, xsec, sum of weights, and whether the sample has WCs.
- `analysis/topeft_run2/analysis_processor.py:1900-1911`: fills
  `process=histAxisName`.

The plotting script later groups these raw process labels into physics groups
using metadata patterns:

- `topeft/params/cr_sr_plots_metadata.yml:432-503`: SR group map, including
  patterns for `ttH`, `ttlnu`, `ttll`, `tllq`, `tHq`, and `tttt`.
- `analysis/topeft_run2/make_cr_and_sr_plots.py:5810-5832`: groups process
  bins and validates labels.

### Category and channel labels

The processor stores selected analysis categories on the `channel` axis. It
also stores application-region labels on the `appl` axis, for example signal
region versus application region.

Source:

- `analysis/topeft_run2/analysis_processor.py:60-99`: resolves category-group
  names into SR or CR category dictionaries.
- `analysis/topeft_run2/analysis_processor.py:1136-1170`: chooses SR and/or CR
  category lists depending on run options.
- `analysis/topeft_run2/analysis_processor.py:1230-1412`: builds packed
  selections, preselection, SR/CR masks, lepton flavor masks, njet masks, and
  application-region masks.
- `analysis/topeft_run2/analysis_processor.py:1638-1703`: builds the concrete
  category dictionary used in the fill loop.
- `analysis/topeft_run2/analysis_processor.py:1771-1817`: loops over category,
  njet label, application region, lepton channel, and lepton flavor.
- `topeft/params/cr_sr_plots_metadata.yml:20-24`: plotting metadata records
  that CR pkls include lepton flavor in channel labels, while SR pkls do not.
- `topeft/params/cr_sr_plots_metadata.yml:116-352`: SR channel aliases and
  leaves.

### Systematics

The `systematic` axis stores nominal and variation labels. The processor loops
over object variations and weight variations:

- `analysis/topeft_run2/analysis_processor.py:642-669`: defines object and
  weight systematic lists.
- `analysis/topeft_run2/analysis_processor.py:716-719`: builds the object
  systematic loop, including nominal.
- `analysis/topeft_run2/analysis_processor.py:1744-1760`: selects nominal,
  object-shifted, or weight-shifted event weights.
- `analysis/topeft_run2/analysis_processor.py:1900-1911`: fills
  `systematic=wgt_fluct`.

Manual comparisons should first list the actual labels in the pkl and only then
compare pairs such as `JESUp` and `JESDown`. Labels are not guaranteed to be
present in tiny runs or in runs without `--do-systs`.

### EFT coefficient storage

For samples with EFT metadata, the processor reads event-level coefficients:

- `analysis/topeft_run2/analysis_processor.py:620-631`: reads
  `events["EFTfitCoefficients"]`, remaps coefficients to the requested WC
  list, and prepares optional squared-weight coefficients.
- `analysis/topeft_run2/analysis_processor.py:1900-1911`: passes
  `eft_coeff=eft_coeffs_cut` into `HistEFT.fill`.
- `topcoffea/topcoffea/modules/quad_fit_tools.py:217-240`: defines the
  coefficient order with `sm` prepended.
- `topcoffea/topcoffea/modules/eft_helper.py:208-266`: remaps coefficient
  arrays when current and target WC lists differ.

If a sample has no EFT coefficients, `HistEFT.fill` defaults to SM-only
coefficients, according to `histEFT.py:214-224`.

### Nominal vs variation content

Nominal and systematic variations are not separate top-level pkl files by
default. They are labels on the `systematic` axis. In schema-v2, 1D nominal
content is split into scalar/EFT sibling keys and any selected scalar second
moment is a separate `_sumw2` key. The processor declares this layout in
`analysis_processor.py:393-479` and fills selected companions in
`analysis_processor.py:2107-2124`.

### Values and variances

For `HistEFT`, call `eval(wc_values)` to get evaluated values. The plotting
script uses this pattern for HistEFT:

- `analysis/topeft_run2/make_cr_and_sr_plots.py:6231-6277`: helper functions
  call `hist_slice.eval({})` for `HistEFT` and use `.values(...)` for ordinary
  histograms.

For sumw2, current analysis outputs use selected scalar `_sumw2` companions
rather than relying only on regular weighted-hist variance storage. Plotting and
datacard utilities validate the applicable nominal/companion relationship.

### Serialization and pickle behavior

Output pkls are gzip-compressed cloudpickle files:

- `topcoffea/topcoffea/modules/utils.py:399-405`: `dump_to_pkl` writes with
  `gzip.open(..., "wb")` and `cloudpickle.dump`.
- `analysis/topeft_run2/run_analysis.py:1715-1765`: writes the final histogram
  dictionary to `<outpath>/<outname>.pkl.gz`.

`HistEFT.__reduce__` reconstructs categorical axes, the user dense axis,
initialization arguments, WC names, and `_dense_hists`:

- `topcoffea/topcoffea/modules/histEFT.py:307-319`.

The plotting script also monkey-patches `SparseHist._read_from_reduce` for
faster loading:

- `analysis/topeft_run2/make_cr_and_sr_plots.py:55-110`.

## 6. EFT parametrization through the analysis pipeline

This section follows the EFT information, not the generic event selection.

### What the stored polynomial means

For each event and each analysis bin, `HistEFT` stores the coefficients of a
quadratic polynomial in the Wilson coefficients. Conceptually, the evaluated
weight is:

```text
yield(WC) = c(sm,sm)
          + sum_i c(wc_i,sm) * wc_i
          + sum_i c(wc_i,wc_i) * wc_i * wc_i
          + sum_{i>j} c(wc_i,wc_j) * wc_i * wc_j
```

The `sm*sm` term is the Standard Model contribution. Terms with one `sm` and
one WC are linear EFT terms. Terms with two non-SM WCs are quadratic or cross
terms. The actual coefficient order is lower triangular after prepending `SM`
to the WC list: `SM*SM`, `wc0*SM`, `wc0*wc0`, `wc1*SM`, `wc1*wc0`,
`wc1*wc1`, and so on. This ordering is implemented in
`topcoffea/topcoffea/modules/quad_fit_tools.py:217-240` and the lower-triangle
helpers in `topcoffea/topcoffea/modules/eft_helper.py:41-99`.

`HistEFT.eval({})` evaluates that polynomial at all WCs equal to zero, so it
returns the SM prediction. A nonzero point such as `eval({"ctG": 1.0})` uses the
same stored coefficient arrays and substitutes the requested WC values.

### Input metadata and WC names

The Run 3 EFT signal sample JSON supplies the sample-level WC order:

- `input_samples/sample_jsons/signal_samples/ND_SRskim2023/ttH_NDSkim_2023.json:1-40`:
  `histAxisName`, `year`, and `WCnames`.
- `input_samples/cfgs/NDSkim_2023_mc_signal_samples_sr.cfg:7-12`: the 2023 SR
  signal CFG lists the same JSON.

`run_analysis.py` loads JSONs directly or through CFG files:

- `analysis/topeft_run2/run_analysis.py:1206-1260`: loads one JSON payload,
  validates required keys, records `histAxisName`, and attaches the active
  redirector prefix.
- `analysis/topeft_run2/run_analysis.py:1279-1314`: parses CFG files, resolving
  JSON paths relative to the CFG file when possible.
- `analysis/topeft_run2/run_analysis.py:1349-1415`: applies the requested year
  filter before processing.
- `analysis/topeft_run2/run_analysis.py:1539-1556`: if `--wc-list` was not
  supplied, aggregates the processor WC list from sample JSON `WCnames`.

That aggregate WC list is passed into `AnalysisProcessor` at
`analysis/topeft_run2/run_analysis.py:1575-1595`.

### Event-level coefficient arrays

The event files provide the per-event `EFTfitCoefficients` branch. The processor
reads and remaps it:

- `analysis/topeft_run2/analysis_processor.py:620-631`: reads
  `events["EFTfitCoefficients"]`; if the sample WC list differs from the
  processor WC list, calls `efth.remap_coeffs(...)`.
- `topcoffea/topcoffea/modules/eft_helper.py:208-266`: `remap_coeffs` prepends
  `SM`, maps old lower-triangle terms into the target WC order, drops omitted
  WCs, and fills coefficients for missing target WCs with zero.

The same event mask used for the dense histogram variable and event weight is
also applied to the EFT coefficient array:

- `analysis/topeft_run2/analysis_processor.py:1771-1817`: applies category,
  njet, application-region, lepton-channel, and lepton-flavor masks.
- `analysis/topeft_run2/analysis_processor.py:1866-1873`: applies any combined
  axis finite-value mask to the weights and `eft_coeffs_cut`.

### Filling schema-v2 nominal sources

One-dimensional analysis histograms are split by source type. Scalar content is
filled into a `SparseHist` sibling and EFT content into a `HistEFT` sibling;
the processor chooses the sibling from sample metadata. A selected `_sumw2`
companion is always scalar.

The fill call passes categorical labels, dense values, nominal or shifted event
weights, and `eft_coeff`:

- `analysis/topeft_run2/analysis_processor.py:1900-1911`: base histogram fill
  uses `weight=weights_flat` and `eft_coeff=eft_coeffs_cut`.
- `analysis/topeft_run2/analysis_processor.py:2107-2124`: a selected `_sumw2`
  companion fills only for the nominal producer path. Its scalar weight squares
  the complete WC=0 event contribution and has no `eft_coeff` payload.

Inside `HistEFT.fill`, dense values and event weights are repeated once per
quadratic term, the coefficient array is flattened, and the event weight is
multiplied into every coefficient before filling the hidden `quadratic_term`
axis. Source: `topcoffea/topcoffea/modules/histEFT.py:197-249`.

If `eft_coeff` is absent or `None`, `HistEFT.fill` uses SM-only coefficients
`[1, 0, 0, ...]` for every event. This behavior matters for non-EFT samples and
for defensive inspection of partial outputs. Source:
`topcoffea/topcoffea/modules/histEFT.py:214-224`.

### Evaluation, systematics, and sumw2

Systematics are not separate EFT polynomials. They are separate categorical
labels on the same `systematic` axis, each filled with its own selected event
weights and object variations:

- `analysis/topeft_run2/analysis_processor.py:642-719`: systematic labels and
  object/weight variation loops.
- `analysis/topeft_run2/analysis_processor.py:1744-1760`: chooses the active
  nominal or shifted weight.

For a nominal SM yield, select `systematic="nominal"` and call `eval({})`. For a
nonzero WC point, call `eval({"wc_name": value})` after the same category,
process, and systematic selections. Unknown WC names raise a `LookupError` in
`HistEFT._wc_for_eval`; source:
`topcoffea/topcoffea/modules/histEFT.py:251-284`.

Selected `_sumw2` companions are scalar SM/WC=0 second moments. The plotter and
datacard utilities validate the relevant nominal/companion relationship; a
future implementation must preserve this contract or migrate consumers in the
same reviewed change:

- `topeft/modules/datacard_tools.py:175-302`: validates base and `_sumw2`
  companions during pkl loading/merging.
- `analysis/topeft_run2/make_cr_and_sr_plots.py:6231-6277`: evaluates HistEFT
  values at the SM point for plotting.

## 7. How the analysis processor fills histograms

### Histogram declaration

`AnalysisProcessor.__init__` resolves the requested histogram list, builds
axes, and creates histograms:

- `analysis/topeft_run2/analysis_processor.py:114-150`: normalizes requested
  histogram names.
- `analysis/topeft_run2/analysis_processor.py:212-236`: creates sparse axes and
  finds available 1D and 2D variables.
- `analysis/topeft_run2/analysis_processor.py:393-431`: creates scalar and/or
  EFT 1D siblings plus policy-selected scalar companions.
- `analysis/topeft_run2/analysis_processor.py:432-479`: creates scalar 2D
  `SparseHist` objects under their base keys, with optional companions.

### Event selection

Selections are built with masks for lepton multiplicity, trigger, Z windows,
b-tag regions, category definitions, application regions, and optional lepton
flavor splitting. The main selection-building block is:

- `analysis/topeft_run2/analysis_processor.py:1014-1020`: lepton-multiplicity
  selections.
- `analysis/topeft_run2/analysis_processor.py:1230-1412`: preselection,
  category, njet, lepton flavor, and `appl` selections.
- `analysis/topeft_run2/analysis_processor.py:1718-1817`: per-histogram,
  per-systematic, per-category fill loop.

### Weights

The base MC event weight is normalized by luminosity, cross section, generator
weight, and sum of weights:

- `analysis/topeft_run2/analysis_processor.py:681-692`.

Additional nominal and systematic weights are added through correction helpers:

- `analysis/topeft_run2/analysis_processor.py:694-711`: prefiring, parton
  shower, scale, and pileup weight setup.
- `analysis/topeft_run2/analysis_processor.py:1135-1226`: category-specific
  lepton SF, fake-factor, flip-rate, and data-driven behavior.
- `analysis/topeft_run2/analysis_processor.py:1744-1760`: chooses which
  weight variation is used for the current systematic label.

### Nominal fills

The nominal fill is just one pass through the same systematic loop with
`wgt_fluct == "nominal"`. The fill payload includes the dense variable,
category labels, application region, process label, systematic label, event
weight, and EFT coefficients when the histogram requires them:

- `analysis/topeft_run2/analysis_processor.py:1900-1911`.

### Systematic fills

Object systematic variations change the event collections before the selection
and dense variable are computed. Weight systematic variations keep the same
object collection but use a shifted weight. The relevant source ranges are:

- `analysis/topeft_run2/analysis_processor.py:766-961`: object variation
  handling for muons, taus, jets, and MET.
- `analysis/topeft_run2/analysis_processor.py:1718-1760`: variable and weight
  choice for each systematic loop.

### EFT fills

`eft_coeffs_cut` is selected with the same event mask as the event weights and
dense variable:

- `analysis/topeft_run2/analysis_processor.py:1771-1817`: category mask and
  `eft_coeffs_cut`.
- `analysis/topeft_run2/analysis_processor.py:1900-1911`: `eft_coeff` is
  included only for histograms marked as requiring EFT.

### CR vs SR behavior

`run_analysis.py` controls CR and SR behavior through `--skip-sr`, `--skip-cr`,
category dictionaries, and histogram-list aliases:

- `analysis/topeft_run2/run_analysis.py:1058-1067`: resolves requested
  category groups.
- `analysis/topeft_run2/run_analysis.py:1081-1175`: expands histogram-list
  aliases such as `ana` and `cr`.
- `analysis/topeft_run2/analysis_processor.py:359-362`: processor flags for
  systematics, lepton flavor splitting, skip SR, and skip CR.
- `analysis/topeft_run2/analysis_processor.py:1136-1170`: selects SR and/or
  CR category dictionaries.

### Run 2 vs Run 3 behavior

The processor marks an event sample as Run 2 or Run 3 from the sample JSON
`year` string:

- `analysis/topeft_run2/analysis_processor.py:450-466`: `is_run2` is true for
  years starting with `201`, and `is_run3` for years starting with `202`.

The runner expands year aliases:

- `analysis/topeft_run2/fullR3_run.sh:165-178`: `run3` expands to
  `2022 2022EE 2023 2023BPix`.
- `analysis/topeft_run2/fullR3_run.sh:226-316`: chooses CFG files by year and
  CR/SR mode.

## 8. `run_cr.sh` as the source-of-truth runner

The source-of-truth student runner is:

```text
analysis/topeft_run2/run_cr.sh
```

Important local behavior:

- `analysis/topeft_run2/run_cr.sh:10-33`: workspace-specific output path,
  chunk size, pkl tag, default histogram variables, years, and category sets.
- `analysis/topeft_run2/run_cr.sh:72-119`: active `run_cr_block` delegates to
  `./fullR3_run.sh` with `--cr`, `--hist-vars`, `--do-systs`, output path,
  category groups, tau analysis, and split lepton flavor.
- `analysis/topeft_run2/run_cr.sh:125-130`: active main loop runs CR jobs over
  the configured years and category sets.
- `analysis/topeft_run2/run_cr.sh:177-220`: commented SR scaffold shows the
  same script family used for SR runs, with `--sr`, `--do-systs`, `--do-np`,
  category groups, and `--all-analysis`.

Do not run this script blindly for a tutorial. As inspected, the active body is
a CR production-style loop over multiple years and category sets. For an SR
tutorial, treat `run_cr.sh` as the source of the command shape, then use
`fullR3_run.sh --dry-run` to source-validate the downstream command before any
real processing.

`fullR3_run.sh` provides the safe dry-run switch and the tutorial input
override:

- `analysis/topeft_run2/fullR3_run.sh:4-28`: usage includes `--dry-run`,
  `--cr`, `--sr`, `--hist-vars`, `--sample-json`, `--cfg-override`, and
  `-p/--outpath`.
- `analysis/topeft_run2/fullR3_run.sh:59-144`: parses command-line options.
- `analysis/topeft_run2/fullR3_run.sh:146-184`: detects user-provided
  chunk-size and output-path overrides, so dry-runs do not print duplicate
  `-p` options.
- `analysis/topeft_run2/fullR3_run.sh:186-192`: requires exactly one of CR or
  SR mode.
- `analysis/topeft_run2/fullR3_run.sh:228-241`: rejects conflicting or missing
  tutorial input override paths.
- `analysis/topeft_run2/fullR3_run.sh:248-257`: forms the output name as
  `<YEAR_LABEL>CRs_<TAG>` or `<YEAR_LABEL>SRs_<TAG>`.
- `analysis/topeft_run2/fullR3_run.sh:306-351`: uses a single JSON/CFG override
  when requested, otherwise uses the production CFG bundle.
- `analysis/topeft_run2/fullR3_run.sh:366-377`: forwards `--hist-vars` as
  `--hist-list`; CR defaults to `cr`, SR defaults to `ana`.
- `analysis/topeft_run2/fullR3_run.sh:379-420`: builds mode-specific
  `run_analysis.py` options, adding the wrapper default output path only when
  the student did not provide `-p/--outpath`.
- `analysis/topeft_run2/fullR3_run.sh:422-428`: prints the `run_analysis.py`
  command and exits before running it when `--dry-run` is present.

## 9. Quick-run tutorial

### Choose a Run 3 EFT signal sample

For the SR tutorial, use:

```text
input_samples/sample_jsons/signal_samples/ND_SRskim2023/ttH_NDSkim_2023.json
```

Why this sample:

- It is directly listed in the 2023 SR signal CFG used by `fullR3_run.sh`:
  `input_samples/cfgs/NDSkim_2023_mc_signal_samples_sr.cfg:7-12`.
- It has `year: "2023"`, `histAxisName: "ttH_private2023"`, and a non-empty
  `WCnames` list:
  `input_samples/sample_jsons/signal_samples/ND_SRskim2023/ttH_NDSkim_2023.json:1-40`.
- It is an SR skim signal sample, matching the SR-oriented tutorial target.

### Level 1: Recommended wrapper path

This is the safe command shape derived from the commented SR block in
`run_cr.sh` and the option parser in `fullR3_run.sh`. The explicit
`--sample-json` option keeps the input to one Run 3 EFT signal JSON without
editing production CFG files. It does not launch the processor because of
`--dry-run`. This is the recommended tutorial path because it keeps the same
wrapper semantics used by the broader Run 3 runner.

```bash
cd /users/apiccine/work/correction-lib/topeft/analysis/topeft_run2

./fullR3_run.sh \
  -y 2023 \
  -t CL007AC_single_ttH_2023_njets \
  -s 1000 \
  --sr \
  --hist-vars njets \
  --sample-json ../../input_samples/sample_jsons/signal_samples/ND_SRskim2023/ttH_NDSkim_2023.json \
  --dry-run \
  --category-groups 2l \
  --all-analysis \
  -p /tmp/cl007ac_histeft_demo \
  -x futures \
  --nworkers 1 \
  --nchunks 1 \
  --pretend \
  --np-postprocess=skip \
  --prefix root://cmsxrootd.crc.nd.edu/
```

Validated dry-run output in this workspace:

```text
OUT_NAME: 2023SRs_CL007AC_single_ttH_2023_njets
Resolved years: 2023
Input override: sample JSON: ../../input_samples/sample_jsons/signal_samples/ND_SRskim2023/ttH_NDSkim_2023.json
Resolved CFGS: ../../input_samples/sample_jsons/signal_samples/ND_SRskim2023/ttH_NDSkim_2023.json
Resolved region: SR
Resolved histogram list: njets
Resolved output path: /tmp/cl007ac_histeft_demo

Running the following command:
python run_analysis.py ../../input_samples/sample_jsons/signal_samples/ND_SRskim2023/ttH_NDSkim_2023.json --years 2023 --hist-list njets --skip-cr --do-systs --do-np -o 2023SRs_CL007AC_single_ttH_2023_njets -s 1000 --category-groups 2l --all-analysis -p /tmp/cl007ac_histeft_demo -x futures --nworkers 1 --nchunks 1 --pretend --np-postprocess=skip --prefix root://cmsxrootd.crc.nd.edu/
```

Option notes:

- `-y 2023`: one Run 3 year, not the full `run3` alias.
- `--sr`: selects SR mode and makes `fullR3_run.sh` choose SR CFG files.
- `--hist-vars njets`: asks for one small histogram variable.
- `--sample-json .../ttH_NDSkim_2023.json`: overrides the default SR CFG bundle
  with one EFT signal JSON. This is the tutorial-only one-sample path.
- `--category-groups 2l`: limits category construction to the 2l SR group.
- `--dry-run`: prints the downstream command and exits before running Python.
- `-x futures --nworkers 1 --nchunks 1 --pretend`: bounded internal
  `run_analysis.py` options, included in the printed command. `--pretend`
  would stop `run_analysis.py` after input discovery if the dry-run guard were
  removed.
- `--np-postprocess=skip`: prevents the SR default `--do-np` from trying to run
  nonprompt post-processing in a one-signal-sample tutorial.
- `--prefix root://cmsxrootd.crc.nd.edu/`: supplies the redirector that the
  source CFG normally provides before listing this sample JSON.
- `-p /tmp/cl007ac_histeft_demo`: a tutorial output path. When `-p` or
  `--outpath` is provided, `fullR3_run.sh` suppresses its default group output
  path, so the reconstructed command contains one output-path option.

Expected pkl path for an authorized real run without `--dry-run` and without
`--pretend`:

```text
/tmp/cl007ac_histeft_demo/2023SRs_CL007AC_single_ttH_2023_njets.pkl.gz
```

This path follows `fullR3_run.sh:248-257` for the output name and
`run_analysis.py:1017-1019` plus `run_analysis.py:1715-1765` for the pkl write.

### Level 2: Lightweight/tutorial helper path

No separate helper script is needed for this cleanup. The
`fullR3_run.sh --sample-json --dry-run` path already prints the exact
one-json command without editing production CFG files, and adding a second
wrapper would duplicate the same command-construction logic. Use Level 1 for
the source-of-truth wrapper path, or Level 3 when you need to inspect or run the
exact `run_analysis.py` command.

### Level 3: Advanced/internal direct run_analysis.py path

This path is for students who want to understand exactly what the wrapper
dry-run reconstructs. It is an advanced/internal path derived from the wrapper
dry-run, not the primary production workflow. Keep `--pretend` for the first
direct run; remove both `--dry-run` from Level 1 and `--pretend` here only after
the real one-sample run is authorized.

```bash
cd /users/apiccine/work/correction-lib/topeft/analysis/topeft_run2

python run_analysis.py \
  ../../input_samples/sample_jsons/signal_samples/ND_SRskim2023/ttH_NDSkim_2023.json \
  --years 2023 \
  --hist-list njets \
  --skip-cr \
  --do-systs \
  --do-np \
  -o 2023SRs_CL007AC_single_ttH_2023_njets \
  -s 1000 \
  --category-groups 2l \
  --all-analysis \
  -p /tmp/cl007ac_histeft_demo \
  -x futures \
  --nworkers 1 \
  --nchunks 1 \
  --pretend \
  --np-postprocess=skip \
  --prefix root://cmsxrootd.crc.nd.edu/
```

This direct command uses exactly one sample JSON and one output-path option.
The explicit `--prefix` is important because the one-json direct path bypasses
the source CFG line that normally supplies the redirector.

### Prove the old CFG-bundle default still resolves

When changing runner options, first dry-run without `--sample-json` to confirm
the production CFG bundle still resolves:

```bash
cd /users/apiccine/work/correction-lib/topeft/analysis/topeft_run2

./fullR3_run.sh \
  -y 2023 \
  -t CL007AC_default_dryrun \
  -s 1000 \
  --sr \
  --hist-vars njets \
  --dry-run \
  --category-groups 2l \
  --all-analysis \
  -p /tmp/cl007ac_default \
  -x futures \
  --nworkers 1 \
  --nchunks 1 \
  --pretend \
  --np-postprocess=skip
```

This dry-run should print the standard 2023 SR CFG bundle:

```text
../../input_samples/cfgs/NDSkim_2023_background_samples.cfg,
../../input_samples/cfgs/NDSkim_2023_data_samples.cfg,
../../input_samples/cfgs/NDSkim_2023_mc_signal_samples_sr.cfg
```

It should also print `Resolved output path: /tmp/cl007ac_default`, and the
reconstructed command should contain exactly one output-path option:
`-p /tmp/cl007ac_default`.

### If you temporarily edit `run_cr.sh` for a broader tutorial

Do not commit such edits unless the analysis conveners request them. The active
file currently runs CR blocks. For an SR tutorial edit, make a temporary local
change modeled on the commented SR scaffold around
`analysis/topeft_run2/run_cr.sh:177-220`, for example:

```diff
- years=(2022 2022EE 2023 2023BPix 2018)
- pkl_base_tag="CR_muonres"
- vars=(invmass tau0Tpt l0ptcorr)
+ years=(2023)
+ pkl_base_tag="CL007AC_SR_tutorial_ttH"
+ vars=(njets)
```

Then use the SR scaffold, add `--dry-run` first, and keep the output path in a
scratch location. The point is to prove command construction before launching
any real processing.

## 10. How `make_cr_and_sr_plots.py` consumes pkl files

The plotting script accepts one or more pkl files and merges them before
plotting:

- `analysis/topeft_run2/make_cr_and_sr_plots.py:7528-7676`: CLI arguments
  include repeated `-f/--pkl-file-path`, pkl list files, output path/name,
  years, region override, variables, workers, merge options, cache options, and
  systematic switches.
- `analysis/topeft_run2/make_cr_and_sr_plots.py:7679-7692`: resolves pkl paths.
- `analysis/topeft_run2/make_cr_and_sr_plots.py:7727-7854`: detects region,
  loads and merges histograms with `load_and_merge_histogram_pkls`, writes an
  optional cached merged pkl, and dispatches plotting.
- `topeft/modules/datacard_tools.py:175-302`: opens each pkl, requires a
  dictionary with string keys, checks base and `_sumw2` companions when
  requested, validates histogram compatibility, and merges matching keys.

Expected schema-v2 pkl structure:

- top-level object: dictionary;
- 1D keys: `<family>__scalar_nominal` and/or `<family>__eft_nominal`, plus
  policy-selected `<family>_sumw2`;
- 2D keys: base `<family>` plus optional `<family>_sumw2`;
- values: scalar `SparseHist`, EFT `HistEFT`, and scalar second-moment
  companions as specified in [Selective sumw2 schema-v2 artifacts](#17-selective-sumw2-schema-v2-artifacts).

Histogram selection and process grouping:

- `analysis/topeft_run2/make_cr_and_sr_plots.py:2365-2479`: prepares variable
  payloads, finds available channels, handles sumw2 histograms, and filters
  process labels.
- `analysis/topeft_run2/make_cr_and_sr_plots.py:732-808`: resolves channel and
  process axis labels.
- `analysis/topeft_run2/make_cr_and_sr_plots.py:5810-5832`: groups process
  bins from metadata patterns.

Category/channel handling:

- `analysis/topeft_run2/make_cr_and_sr_plots.py:1738-1767`: integrates
  application region and category labels.
- `topeft/modules/yield_tools.py:475-513`: integrates categories and `appl`
  labels for yield extraction.

Systematics:

- `analysis/topeft_run2/make_cr_and_sr_plots.py:6140-6214`: discovers
  systematic labels, completes Up/Down pairs, integrates nominal and variation
  histograms, and prepares arrays.

Yields and HistEFT evaluation:

- `analysis/topeft_run2/make_cr_and_sr_plots.py:6231-6277`: evaluates HistEFT
  with `eval({})` for SM values and uses regular `.values(...)` for other
  histogram types.
- `topeft/modules/yield_tools.py:548-567`: yield code integrates categories,
  evaluates HistEFT at a requested WC point, and combines value and variance
  arrays.

Manual inspectors must reproduce these assumptions: dictionary pkl, string
keys, base and `_sumw2` pairing, categorical axes named `process`, `channel`,
`systematic`, and `appl`, and explicit HistEFT evaluation before using values
as physics yields.

## 11. Manual pkl inspection

### Use the helper script

Before running these commands, activate the analysis environment you normally
use for `topeft` and `topcoffea`. Codex validation uses a workspace wrapper in
reports, but students should not copy that machinery into ordinary inspection
commands.

The helper is read-only:

```bash
cd /users/apiccine/work/correction-lib/topeft
python analysis/topeft_run2/inspect_histeft_pkl.py --help
```

Inspect a pkl:

```bash
python analysis/topeft_run2/inspect_histeft_pkl.py /path/to/output.pkl.gz --max-labels 10
```

Inspect one histogram and ask for a simple nominal total when discoverable:

```bash
python analysis/topeft_run2/inspect_histeft_pkl.py /path/to/output.pkl.gz --hist njets --max-labels 10 --yield-summary
```

The helper prints:

- top-level object type;
- top-level keys;
- histogram-like object types;
- axes and labels;
- `process`, `channel`, `systematic`, and `appl` labels when discoverable;
- WC names when the object exposes them;
- optional simple nominal yield and variance sums.

### Minimal manual Python snippets

Use the same analysis environment when opening analysis pkls. A pkl may require
`topcoffea` classes to be importable.

List top-level keys:

```bash
python -c 'import gzip,pickle; p="/path/to/output.pkl.gz"; f=gzip.open(p,"rb") if p.endswith(".gz") else open(p,"rb"); obj=pickle.load(f); print(type(obj)); print(list(obj)[:20])'
```

List axes for one histogram:

```bash
python -c 'import gzip,pickle; p="/path/to/output.pkl.gz"; obj=pickle.load(gzip.open(p,"rb")); h=obj["njets"]; print(type(h)); print([ax.name for ax in h.axes])'
```

List labels on common categorical axes:

```bash
python -c 'import gzip,pickle; p="/path/to/output.pkl.gz"; h=pickle.load(gzip.open(p,"rb"))["njets"]; print(list(h.axes["process"])[:20]); print(list(h.axes["channel"])[:20]); print(list(h.axes["systematic"])[:20])'
```

Evaluate a HistEFT at the SM point and sum all returned blocks:

```bash
python -c 'import gzip,pickle,numpy as np; p="/path/to/output.pkl.gz"; h=pickle.load(gzip.open(p,"rb"))["njets"]; vals=h.integrate("systematic","nominal").eval({}); print(sum(float(np.nansum(v)) for v in vals.values()))'
```

Compare nominal to one systematic label:

```bash
python -c 'import gzip,pickle,numpy as np; p="/path/to/output.pkl.gz"; h=pickle.load(gzip.open(p,"rb"))["njets"]; nom=h.integrate("systematic","nominal").eval({}); up=h.integrate("systematic","JESUp").eval({}); print(sum(float(np.nansum(up[k]-nom.get(k,0))) for k in up))'
```

The last snippet assumes `JESUp` exists. Always list systematic labels first.

### Make a small yield table

For a quick manual table, select one histogram, integrate one systematic label,
then loop over process labels:

```bash
python -c 'import gzip,pickle,numpy as np; p="/path/to/output.pkl.gz"; h=pickle.load(gzip.open(p,"rb"))["njets"].integrate("systematic","nominal"); procs=list(h.axes["process"]); print("process yield"); [print(proc, sum(float(np.nansum(v)) for v in h.integrate("process",proc).eval({}).values())) for proc in procs[:20]]'
```

This is deliberately simple. It does not group processes, handle overflow
policy, or combine sumw2 uncertainties the same way the plotter does. Use it as
a first sanity check, not as a publication number.

### Check EFT/WC content

For a `HistEFT`, inspect WC names:

```bash
python -c 'import gzip,pickle; p="/path/to/output.pkl.gz"; h=pickle.load(gzip.open(p,"rb"))["njets"]; print(getattr(h,"wc_names", getattr(h,"_wc_names", None)))'
```

Evaluate a non-SM point if the WC exists:

```bash
python -c 'import gzip,pickle,numpy as np; p="/path/to/output.pkl.gz"; h=pickle.load(gzip.open(p,"rb"))["njets"].integrate("systematic","nominal"); vals=h.eval({"ctG": 1.0}); print(sum(float(np.nansum(v)) for v in vals.values()))'
```

If this raises a `LookupError`, the histogram WC list does not include that WC.

## 12. Debugging checklist

Missing pkl:

- Check whether the command was a dry-run or used `--pretend`; those do not
  write the final pkl.
- Check the final `-p` output path and `-o` output name in the printed
  `run_analysis.py` command.
- Check `run_analysis.py:1017-1019` and `run_analysis.py:1715-1765` for output
  path construction and writing.

Empty histograms:

- Verify the category group exists in the active `ch_lst.json` block.
- Check whether `--skip-sr` or `--skip-cr` removed the target region.
- List `channel`, `process`, and `systematic` labels with the helper.
- Use a broad category first, then narrow.

Missing category:

- Confirm the category group passed through `--category-groups`.
- Check `analysis_processor.py:60-99` for group resolution and
  `analysis_processor.py:1638-1703` for concrete category labels.
- For plotting, check `topeft/params/cr_sr_plots_metadata.yml:20-24` because
  CR and SR channel-label expectations differ.

Missing process/sample:

- Confirm the sample JSON is present in the CFG selected by `fullR3_run.sh`.
- Check the sample JSON `histAxisName`; this is the raw process-axis label.
- Check plot grouping patterns in `topeft/params/cr_sr_plots_metadata.yml`.

Missing systematic:

- Confirm the run used `--do-systs`.
- Tiny or pretend runs may not fill all requested variations.
- List actual `systematic` labels before comparing Up/Down pairs.

Missing EFT coefficients:

- Check the sample JSON `WCnames` list.
- Check whether event files have `EFTfitCoefficients`.
- Check `analysis_processor.py:620-631` for the coefficient read/remap path.
- If `HistEFT.eval({"ctG": 1.0})` fails, list the histogram's WC names.

Mismatched histogram variable names:

- `fullR3_run.sh --hist-vars` forwards to `run_analysis.py --hist-list`.
- Valid 1D names are in `topeft/modules/axes.py:1-228`.
- Valid 2D names are in `topeft/modules/axes.py:230-260` and later entries.

CR/SR mismatch:

- `--cr` makes `fullR3_run.sh` add `--skip-sr`.
- `--sr` makes `fullR3_run.sh` add `--skip-cr`.
- The active `run_cr.sh` body is CR-oriented in this workspace.

Wrong sample JSON/CFG:

- For 2023 SR, `fullR3_run.sh` uses:
  `NDSkim_2023_background_samples.cfg`,
  `NDSkim_2023_data_samples.cfg`, and
  `NDSkim_2023_mc_signal_samples_sr.cfg`.
- The chosen tutorial sample is listed in
  `input_samples/cfgs/NDSkim_2023_mc_signal_samples_sr.cfg:9`.

Pkl too large to inspect naively:

- Use the helper with `--hist` and small `--max-labels`.
- Avoid printing full `.values()` arrays.
- Start with top-level keys and axes.
- For many pkls, use the plotter merge-only path or a small custom inspector
  before loading every histogram into plotting.

## 13. Detailed HistEFT API reference

This section documents the public and practically relevant `HistEFT` API as it
exists now. It is source-grounded in
`topcoffea/topcoffea/modules/histEFT.py` and inherited behavior from
`topcoffea/topcoffea/modules/sparseHist.py`.

### Constructor and core attributes

`HistEFT(*args, wc_names=None, **kwargs)`

Source: `topcoffea/topcoffea/modules/histEFT.py:74-126`.

- `*args`: histogram axes. All axes must be named. Categorical axes should come
  first and use growth categories. The last user axis must be the one physics
  dense axis and must be `Regular`, `Variable`, or `Integer`.
- `wc_names`: list of Wilson coefficient names, without `SM`. If omitted or
  falsey, the histogram has no non-SM WCs and stores only the SM coefficient
  term.
- `storage`: accepted through `kwargs`, but only `"Double"` is supported. If not
  supplied, the constructor sets it to `"Double"`.
- `rebin`: accepted in `kwargs` but rejected when true; current HistEFT does not
  implement rebinning.
- `quadratic_term`: if the last supplied axis is named `quadratic_term`, HistEFT
  uses it as the coefficient axis. Otherwise it creates a hidden
  `hist.axis.Integer(start=0, stop=n_quad_terms, name="quadratic_term")`.

Important attributes:

- `_wc_names`: ordered mapping from WC name to WC index.
- `_wc_count`: number of non-SM WCs.
- `_quad_count`: number of stored quadratic terms, from
  `efth.n_quad_terms(n_wc)` at `eft_helper.py:41-46`.
- `_coeff_axis`: hidden `quadratic_term` axis.
- `_dense_axis`: user physics dense axis.
- `_dense_hists`: inherited sparse storage dictionary mapping categorical keys
  to dense `hist.Hist` blocks.
- `_init_args_eft`: stores `wc_names` for copy/pickle reconstruction.

Reserved axis names are `quadratic_term`, `sample`, `weight`, and `thread`.
These are rejected at construction time.

### WC metadata helpers

`wc_names`

Source: `histEFT.py:133-135`.

Returns a list of WC names in the histogram order. The current implementation
builds this from `_wc_names`; students should treat the returned order as the
evaluation order for array-style WC values.

`index_of_wc(wc)`

Source: `histEFT.py:137-138`.

Returns the integer index of one WC in `_wc_names`. Unknown names propagate a
`KeyError`. This is mainly a helper for `quadratic_term_index`.

`quadratic_term_index(*wcs)`

Source: `histEFT.py:140-163`.

Takes exactly two coefficient names and returns the lower-triangle coefficient
index. `"sm"` maps to 0; non-SM names map to `index_of_wc(name) + 1`. The method
orders the two factors internally, so `("sm", "ctG")` and `("ctG", "sm")`
resolve to the same term. This method is public enough that a replacement should
reproduce it unless all downstream coefficient-axis inspection is migrated.

### Fill API

`fill(eft_coeff=None, **values)`

Source: `histEFT.py:197-249`.

Expected keyword payload in this analysis:

- scalar categorical labels: `process`, `channel`, `systematic`, and `appl`;
- one dense variable array whose keyword is the physics variable name;
- `weight`: one event weight per selected event;
- `eft_coeff`: array shaped like `(n_events, n_quad_terms)`.

If `eft_coeff` is `None`, HistEFT broadcasts SM-only coefficients
`[1, 0, 0, ...]` for every event. During filling, the dense variable and event
weight are repeated once per quadratic term, the coefficient array is flattened,
and `weight` is multiplied into every coefficient. Then the inherited
`SparseHist.fill` receives the repeated dense values, the repeated
`quadratic_term` indices, scalar categories, and the flattened weighted
coefficients as the dense histogram weight.

The method annotation says it returns `Self`, but the implementation delegates
to `super().fill(...)` without returning that result. Current processor code
does not rely on a return value. A drop-in replacement should either preserve
this benign behavior or audit all callers before changing it.

Internal helpers used by `fill`:

- `_fill_flatten(a, n_events)`, source `histEFT.py:172-188`: accepts scalar-like,
  1D, or `(n_events, 1)` arrays and repeats event values over quadratic terms.
  It raises `ValueError` for incompatible dimensions.
- `_fill_indices(n_events)`, source `histEFT.py:190-195`: creates repeated
  `quadratic_term` indices for all events.

### Evaluation API

`eval(values)`

Source: `histEFT.py:271-284`.

Evaluates every populated categorical block at one WC point and returns a
dictionary from sparse categorical key tuples to NumPy arrays over the user
dense axis, including flow bins. Accepted `values` are:

- `None`: all WCs zero;
- mapping such as `{"ctG": 1.0}`: unspecified WCs are zero;
- array-like: interpreted in `wc_names` order.

Internally, `eval` calls `self.view(flow=True, as_dict=True)` and passes the
stored coefficient arrays, excluding the coefficient-axis flow columns with
`hvs[..., 1:-1]`, to `efth.calc_eft_weights(...)`.

`_wc_for_eval(values)`

Source: `histEFT.py:251-269`.

Normalizes mapping, array, or `None` inputs into a WC-value array. Unknown WC
names raise `LookupError` with the known coefficient list. Plotting and manual
inspection rely on this error being clear.

`as_hist(values)`

Source: `histEFT.py:286-305`.

Evaluates the histogram and materializes a regular `hist.Hist` without the
hidden `quadratic_term` axis. It is useful for plotting or for APIs that need a
normal histogram after choosing one WC point.

`calc_eft_weights(q_coeffs, wc_values)`

Source: `histEFT.py:358-388`.

Local method equivalent in spirit to `eft_helper.calc_eft_weights`, but adjusted
for coefficient-axis flow columns. The comment says it should move to
`eft_helper` once HistEFT is replaced. Current `eval` uses
`efth.calc_eft_weights(...)`, so treat this method as a compatibility/internal
helper unless a caller is found.

### Values, variances, and views

`HistEFT` does not define its own `variances(...)` method. The current analysis
uses base histograms plus `_sumw2` companion histograms for uncertainty
handling. A manual inspector should evaluate the base HistEFT for yields and use
the matching `_sumw2` key when it needs the squared-weight convention used by
the plotter/datacard code.

Inherited from `SparseHist`:

- `values(flow=False)`, source `sparseHist.py:327-350`: recursively returns dense
  histogram `.values(...)` arrays over populated categorical keys as an awkward
  structure. For raw HistEFT, this exposes stored coefficient-axis content, not
  evaluated physics yields.
- `counts(flow=False)`, source `sparseHist.py:352-353`: delegates to dense
  histogram counts.
- `view(flow=False, as_dict=True)`, source `sparseHist.py:362-371`: returns a
  dictionary from categorical key tuple to dense histogram view. HistEFT
  evaluation depends on `as_dict=True`.

### Slicing, integration, grouping, and category management

Inherited practical API from `SparseHist`:

- `__getitem__(key)`, source `sparseHist.py:299-325`: supports mapping-style
  selection such as `h[{"systematic": "nominal"}]`. Depending on what axes
  collapse, it returns a new sparse histogram, a dense `hist.Hist`, or a scalar.
- `__setitem__(key, value)`, source `sparseHist.py:273-297`: assigns into one
  categorical key/dense selection. Plotting code can rely on assignment when
  constructing derived histograms.
- `integrate(name, value=None)`, source `sparseHist.py:373-376`: implemented as
  slicing; `value=None` means sum over that axis.
- `group(axis_name, groups)`, source `sparseHist.py:378-406`: creates a new
  sparse histogram where selected categorical bins are merged into named groups.
  Process grouping in plotting depends on this behavior.
- `remove(axis_name, bins)` and `prune(axis, to_keep)`, source
  `sparseHist.py:408-433`: remove categorical labels or keep only a subset.
- `categorical_axes`, `dense_axes`, and `categorical_keys`, source
  `sparseHist.py:111-122`: expose sparse/dense axis metadata and populated
  categorical keys.

### Copy, identity, arithmetic, and merging

Inherited from `SparseHist`:

- `empty_from_axes(...)`, source `histEFT.py:128-131` and `sparseHist.py:60-70`:
  creates an empty histogram of the same class, preserving `wc_names`.
- `__copy__` and `__deepcopy__`, source `sparseHist.py:75-83`: copy empty or
  populated sparse histograms.
- `reset()` and `empty()`, source `sparseHist.py:355-443`: reset dense blocks and
  test whether all dense views are zero.
- `scale(factor)`, source `sparseHist.py:435-437`: in-place scalar
  multiplication and returns `self`.
- arithmetic methods, source `sparseHist.py:445-522`: in-place and out-of-place
  addition/multiplication/division delegate to dense histograms. SparseHist merge
  requires categorical axis names and order to match.
- `identity()`, source `sparseHist.py:524-529`: deprecated old-coffea
  compatibility helper returning an empty copy.

These methods are important for coffea accumulation, pkl merging, and plotting.
A drop-in replacement must support compatible addition and in-place addition at
minimum.

### Pickle compatibility

`HistEFT.__reduce__`

Source: `histEFT.py:307-319`.

Pickle state contains categorical axes, the user dense axis, initialization
arguments including `wc_names`, and `_dense_hists`. `_read_from_reduce` delegates
to `SparseHist` reconstruction at `histEFT.py:354-356` and
`sparseHist.py:477-483`.

The plotting script installs a compatibility fast loader for `SparseHist`
reduce state:

- `analysis/topeft_run2/make_cr_and_sr_plots.py:55-110`.

Future replacements need a documented serialization story: either old HistEFT
pkls remain readable, or a converter and a compatibility boundary must be
provided before production outputs are migrated.

### Public API versus implementation detail

Treat these as current public or practically public:

- constructor call pattern;
- `wc_names`;
- `quadratic_term_index`;
- `fill(eft_coeff=..., weight=..., <axis names>=...)`;
- `eval`, `as_hist`;
- `.axes`, `.integrate`, `.group`, `.remove`, `.prune`, `.values`, `.view`;
- arithmetic/add/merge behavior;
- pickle load/save compatibility.

Treat these as implementation details unless a caller is found:

- `_wc_names`, `_quad_count`, `_coeff_axis`, `_dense_hists`;
- `_fill_flatten`, `_fill_indices`, `_wc_for_eval`;
- the local `HistEFT.calc_eft_weights` method;
- exact internal awkward layout from `SparseHist.values`.

The future replacement can hide or redesign the internal pieces only if the
processor, plotter, yield tools, datacard tools, and pkl compatibility plan are
updated together.

## 14. Mapping to a future scikit-hist EFT-aware replacement

A future `scikit-hist` EFT-aware histogram class must reproduce current
behavior before it can replace `HistEFT` safely.

Core histogram behavior:

- accept named categorical axes and dense axes;
- preserve sparse categorical behavior or provide an equivalent memory-safe
  representation;
- preserve the current constructor call shape from `analysis_processor.py:245-292`
  or provide a small adapter with identical behavior;
- support fill calls with scalar categorical labels, dense arrays, event
  weights, and optional EFT coefficient arrays;
- preserve the current "one user dense axis plus hidden EFT coefficient axis"
  semantics unless the processor fill path is migrated at the same time;
- support addition and in-place addition for merging pkl outputs;
- support pruning, removal, grouping, slicing, projection, and integration
  patterns used by the processor and plotter.

Axes and metadata behavior:

- preserve axis names: `process`, `channel`, `systematic`, `appl`, dense
  variable name, and EFT coefficient dimension;
- preserve process labels from sample JSON `histAxisName`;
- preserve channel labels exactly as produced by the processor;
- preserve systematic labels exactly as filled;
- keep WC names and their order available after pickle load;
- preserve enough metadata for plotting group maps and yield tools.

Values and variances behavior:

- expose SM and WC-evaluated values with clear flow-bin policy;
- preserve or replace the current `_sumw2` companion convention;
- make variance behavior explicit, because current plotting often treats
  `<hist>_sumw2` as the variance source rather than using only weighted storage;
- support efficient summed yields over process/channel/systematic selections.

EFT coefficient storage and evaluation:

- store the quadratic coefficient order currently produced by
  `quad_fit_tools.py:217-240`;
- reproduce `HistEFT.quadratic_term_index` behavior from
  `histEFT.py:140-163`;
- reproduce `fill(eft_coeff=...)` semantics from `histEFT.py:197-249`,
  including SM-only default coefficients for non-EFT samples;
- reproduce `eval({})`, `eval({"wc": value})`, and unknown-WC error behavior;
- support coefficient remapping or define a stricter common WC-list contract;
- keep coefficient evaluation independent of process/channel/systematic slicing,
  because the current plotter first slices/group/integrates and then evaluates
  the selected HistEFT object at the SM point.

Systematic variation handling:

- allow `systematic` to remain a categorical axis;
- support nominal and Up/Down comparisons without forcing all variations to be
  dense axes;
- preserve missing-variation behavior for small or partial runs.

Processor API assumptions:

- constructor can be called where `analysis_processor.py:245-292` currently
  calls `HistEFT(...)`;
- `fill` accepts `process=`, `channel=`, `systematic=`, `appl=`, dense variable
  keyword, `weight=`, and `eft_coeff=`;
- `fill` tolerates `eft_coeff=None` by filling SM-only coefficients;
- event weights multiply EFT coefficients during fill, not during later
  evaluation;
- histograms are pickleable and mergeable after coffea execution;
- 2D non-EFT histograms can remain separate if the first replacement only
  targets 1D EFT-aware histograms.

Plotting API assumptions:

- producer pkl keys follow the schema-v2 sibling layout; any uniform variable
  names in a plotter are a consumer-local materialized view;
- objects expose `.axes`, `.integrate(...)`, `.group(...)`, `.remove(...)`,
  `.prune(...)`, `.values(...)`, and HistEFT-like `.eval(...)`;
- process grouping and channel integration work with existing labels;
- `load_and_merge_histogram_pkls` compatibility checks can validate axes,
  dense binning, and WC metadata;
- `eval({})` returns a dictionary of arrays over dense bins, including a
  flow-bin policy compatible with `_values_with_flow_or_overflow` in
  `make_cr_and_sr_plots.py:6231-6277`.

Serialization compatibility:

- old pkls may need a compatibility reader;
- new pkls should load without monkey-patches;
- pkl size and load time matter because plotting merges large outputs;
- a migration may need a converter from old HistEFT pkls to the new format.

What can be simplified if plotting migrates too:

- the new class does not need to mimic every legacy `SparseHist` method if the
  plotter and yield tools move to a clearer shared API at the same time;
- process grouping can be centralized outside histogram objects;
- sumw2 handling can be made explicit as variance storage instead of separate
  top-level keys;
- WC evaluation can return regular `hist` or `scikit-hist` objects with a
  documented flow-bin convention;
- old pkl support can be isolated in a converter rather than in the new runtime
  class, if the migration plan includes converting or regenerating existing
  outputs.

Tests to write before swapping implementation:

CL007AB and CL007AC document these requirements at tutorial level. The formal
implemented-vs-used matrix and parity-test specification now live in
[`docs/histeft_api_contract.md`](histeft_api_contract.md).

- fill one EFT sample and one non-EFT sample, then compare SM yields;
- compare `eval({})` and several nonzero WC points against current `HistEFT`;
- verify category, process, systematic, and `appl` labels after pickle round
  trip;
- verify `_sumw2` or replacement variance behavior;
- verify merge/add behavior for two compatible pkls;
- verify plotting path on a small SR and CR pkl;
- verify failure messages for unknown WCs, incompatible axes, and missing
  sumw2 companions.

## 15. Glossary

`HistEFT`
: EFT-aware histogram class implemented in `topcoffea`. Stores quadratic EFT
  coefficient terms and evaluates bin contents at a chosen WC point.

`SparseHist`
: Sparse categorical histogram layer used by `HistEFT`. Stores dense histogram
  blocks only for populated categorical keys.

`WC`
: Wilson coefficient.

`EFTfitCoefficients`
: Event branch containing quadratic coefficient values used by `HistEFT.fill`.

`process`
: Histogram axis label usually sourced from sample JSON `histAxisName`.

`channel`
: Histogram axis label for analysis category, sometimes including lepton flavor
  and njet suffixes.

`systematic`
: Histogram axis label for nominal and systematic variations.

`appl`
: Histogram axis label for signal/application-region selection.

`_sumw2`
: Companion histogram key convention for storing squared-weight content.

`CR`
: Control region.

`SR`
: Signal region.

`CFG`
: Text file listing sample JSON files and optional redirector prefixes.

## 16. Source map: relevant files and line ranges

HistEFT and sparse histogram implementation:

- `topcoffea/topcoffea/modules/histEFT.py:23-72`: class purpose and examples.
- `topcoffea/topcoffea/modules/histEFT.py:74-126`: constructor restrictions,
  WC metadata, and `quadratic_term` axis.
- `topcoffea/topcoffea/modules/histEFT.py:140-163`: quadratic-term indexing.
- `topcoffea/topcoffea/modules/histEFT.py:172-195`: fill-shape helpers.
- `topcoffea/topcoffea/modules/histEFT.py:197-249`: EFT-aware fill.
- `topcoffea/topcoffea/modules/histEFT.py:251-269`: WC-value normalization.
- `topcoffea/topcoffea/modules/histEFT.py:271-305`: `eval` and `as_hist`.
- `topcoffea/topcoffea/modules/histEFT.py:307-319`: pickle reduce state.
- `topcoffea/topcoffea/modules/histEFT.py:321-388`: scaling helper and local
  EFT-weight evaluator.
- `topcoffea/topcoffea/modules/sparseHist.py:15-39`: sparse/dense axis model.
- `topcoffea/topcoffea/modules/sparseHist.py:124-139`: fill bookkeeping.
- `topcoffea/topcoffea/modules/sparseHist.py:273-325`: assignment and slicing
  behavior.
- `topcoffea/topcoffea/modules/sparseHist.py:349-406`: values, view,
  integrate, and group.
- `topcoffea/topcoffea/modules/sparseHist.py:408-529`: remove, prune, scale,
  arithmetic, pickle reconstruction, and identity.

EFT helpers and pkl helpers:

- `topcoffea/topcoffea/modules/quad_fit_tools.py:203-240`: coefficient
  extraction and ordering.
- `topcoffea/topcoffea/modules/eft_helper.py:9-46`: EFT polynomial evaluation
  and quadratic-term count.
- `topcoffea/topcoffea/modules/eft_helper.py:80-99`: lower-triangle quadratic
  term/factor mapping.
- `topcoffea/topcoffea/modules/eft_helper.py:132-206`: quartic/squared-weight
  coefficient helpers.
- `topcoffea/topcoffea/modules/eft_helper.py:208-266`: coefficient remapping.
- `topcoffea/topcoffea/modules/utils.py:399-405`: pkl writing helper.
- `topcoffea/topcoffea/modules/compat.py:13-39`: HistEFT pickle compatibility
  hook used by local yield tools and the inspector.

Processor:

- `analysis/topeft_run2/analysis_processor.py:1-30`: imports `HistEFT`.
- `analysis/topeft_run2/analysis_processor.py:60-99`: category group
  resolution.
- `analysis/topeft_run2/analysis_processor.py:112-158`: processor init and
  histogram-name normalization.
- `analysis/topeft_run2/analysis_processor.py:212-343`: histogram axes and
  declarations.
- `analysis/topeft_run2/analysis_processor.py:450-466`: sample metadata.
- `analysis/topeft_run2/analysis_processor.py:620-631`: EFT coefficient read
  and remap.
- `analysis/topeft_run2/analysis_processor.py:642-719`: systematic lists.
- `analysis/topeft_run2/analysis_processor.py:681-711`: nominal weight setup.
- `analysis/topeft_run2/analysis_processor.py:1135-1226`: category-specific
  weights and data-driven behavior.
- `analysis/topeft_run2/analysis_processor.py:1230-1412`: selections.
- `analysis/topeft_run2/analysis_processor.py:1414-1629`: dense variables.
- `analysis/topeft_run2/analysis_processor.py:1638-1703`: category dictionary.
- `analysis/topeft_run2/analysis_processor.py:1718-1924`: fill loop and sumw2
  fill.

Runner:

- `analysis/topeft_run2/run_cr.sh:10-33`: local runner defaults.
- `analysis/topeft_run2/run_cr.sh:72-119`: active CR block.
- `analysis/topeft_run2/run_cr.sh:125-130`: active main CR loop.
- `analysis/topeft_run2/run_cr.sh:177-220`: commented SR scaffold.
- `analysis/topeft_run2/fullR3_run.sh:4-28`: usage and options.
- `analysis/topeft_run2/fullR3_run.sh:59-144`: option parsing.
- `analysis/topeft_run2/fullR3_run.sh:146-184`: pass-through chunk-size and
  output-path override detection.
- `analysis/topeft_run2/fullR3_run.sh:186-241`: CR/SR, year, and input-override
  validation.
- `analysis/topeft_run2/fullR3_run.sh:248-257`: output-name construction.
- `analysis/topeft_run2/fullR3_run.sh:261-351`: CFG selection and
  `--sample-json`/`--cfg-override` handling.
- `analysis/topeft_run2/fullR3_run.sh:366-428`: hist-list forwarding,
  command construction, and dry-run exit.
- `analysis/topeft_run2/run_analysis.py:639-860`: CLI arguments.
- `analysis/topeft_run2/run_analysis.py:1017-1051`: output paths and test
  mode.
- `analysis/topeft_run2/run_analysis.py:1058-1175`: category and histogram-list
  resolution.
- `analysis/topeft_run2/run_analysis.py:1177-1463`: JSON/CFG loading and sample
  setup.
- `analysis/topeft_run2/run_analysis.py:1532-1556`: pretend mode and WC-list
  aggregation.
- `analysis/topeft_run2/run_analysis.py:1575-1595`: processor construction.
- `analysis/topeft_run2/run_analysis.py:1678-1765`: runner execution and pkl
  writing.

Run 3 EFT signal sample:

- `input_samples/cfgs/NDSkim_2023_mc_signal_samples_sr.cfg:7-12`: 2023 SR EFT
  signal JSON list.
- `input_samples/sample_jsons/signal_samples/ND_SRskim2023/ttH_NDSkim_2023.json:1-40`:
  selected tutorial sample metadata and WC names.

Plotting and yield consumers:

- `analysis/topeft_run2/make_cr_and_sr_plots.py:55-110`: SparseHist pickle
  load patch.
- `analysis/topeft_run2/make_cr_and_sr_plots.py:732-808`: process/channel axis
  helpers.
- `analysis/topeft_run2/make_cr_and_sr_plots.py:1738-1767`: category and appl
  integration.
- `analysis/topeft_run2/make_cr_and_sr_plots.py:2365-2479`: variable payload
  preparation.
- `analysis/topeft_run2/make_cr_and_sr_plots.py:5810-5832`: process grouping.
- `analysis/topeft_run2/make_cr_and_sr_plots.py:6140-6214`: systematic
  extraction.
- `analysis/topeft_run2/make_cr_and_sr_plots.py:6231-6277`: HistEFT value
  evaluation.
- `analysis/topeft_run2/make_cr_and_sr_plots.py:7528-7854`: plotter CLI,
  loading, merging, and dispatch.
- `topeft/modules/datacard_tools.py:175-302`: pkl loading and merge validation.
- `topeft/modules/yield_tools.py:305-383`: axis and category label helpers.
- `topeft/modules/yield_tools.py:475-567`: category/appl integration and yield
  extraction.

Plot metadata:

- `topeft/params/cr_sr_plots_metadata.yml:20-24`: CR/SR channel-label
  convention.
- `topeft/params/cr_sr_plots_metadata.yml:116-352`: SR channels.
- `topeft/params/cr_sr_plots_metadata.yml:432-503`: SR process group patterns.
- `topeft/params/cr_sr_plots_metadata.yml:530-554`: Run 2 and Run 3 lumi
  metadata.

## 17. Selective sumw2 schema-v2 artifacts

Current selective-sumw2 output has nominal container schema version 2 and the
`split_sibling_v1` layout. For a 1D family, inspect these exact top-level keys:

```text
<family>__scalar_nominal   scalar SparseHist source, when scalar content exists
<family>__eft_nominal      HistEFT source, when EFT content exists
<family>_sumw2             selected scalar second-moment companion, when selected
```

The scalar and EFT nominal siblings have non-overlapping process content. A
family need not have all three keys: an empty content class has no nominal
sibling, and a companion appears only when the resolved storage policy selects
it. The old unsplit 1D `<family>` producer key is absent in schema-v2 output.

Two-dimensional families do not split: they remain scalar `SparseHist` objects
under `<family>` and may have an optional `<family>_sumw2` companion. Do not
look for `__scalar_nominal` or `__eft_nominal` keys for a 2D family.

```text
run_analysis.py
    -> schema-v2 pkl (split 1D sources; selected scalar companions)
    -> data-driven processing (uses required scalar companions; AR content is transient)
    -> plotter / DatacardMaker consumer-local materialization
    -> final plotting or exact-SR datacard selection
```

The materialized mapping at the last boundary is a consumer-local view, not a
new producer layout. Start inspection by listing top-level keys, reading the
artifact provenance/schema metadata, and validating family and companion names
before filtering or merging. Keep inspection read-only; a filename or output
tag does not establish schema validity. Exact selection, WC=0 second-moment,
merge, and collision semantics are maintained in the
[HistEFT API contract](histeft_api_contract.md#15-selective-sumw2-schema-and-consumer-contract).

## 18. Inspecting data-driven applicability

The maintained sidecar records which data-driven products are enabled by the
source-wide policy and which products are applicable to each family. Read the
sidecar through the maintained artifact helpers; do not infer applicability
from a filename or edit the JSON by hand. The normative definitions and
versioning procedure are in the [API contract](histeft_api_contract.md#16-data-driven-applicability-and-transformed-artifacts).

For a transformed artifact, this read-only example inspects one family and
then streams only its statistical companion. It uses snake_case names and does
not retain the full pickle in memory:

```python
from topcoffea.modules.hist_utils import iterate_hist_from_pkl
from topeft.modules.histogram_artifact import (
    read_histogram_sidecar,
    validate_histogram_artifact,
)

input_pkl = "/path/to/processor_np.pkl.gz"
family_name = "l0eta"

sidecar = read_histogram_sidecar(input_pkl)
validated_artifact = validate_histogram_artifact(input_pkl)
assert validated_artifact["metadata"] == sidecar
family_contract = sidecar["transformation_contract"]["families"][family_name]
family_manifest = sidecar["sumw2_content_manifest"]["families"][family_name]

print(family_contract["source_application_regions"])
print(family_contract["applicable_products"])
print(family_contract["generated_nonprompt_processes"])
print(family_contract["generated_flips_processes"])
print(family_manifest["required_sumw2_processes"])

actual_sumw2_processes = None
for key, histogram in iterate_hist_from_pkl(
    input_pkl, allow_empty=False, materialize=False
):
    if key == f"{family_name}_sumw2":
        actual_sumw2_processes = sorted(
            str(process) for process in histogram.axes["process"]
        )
        break

required_sumw2_processes = sorted(
    family_manifest["required_sumw2_processes"]
)
assert actual_sumw2_processes == required_sumw2_processes
```

`read_histogram_sidecar(input_pkl)` is useful when only the validated metadata
is needed; `validate_histogram_artifact(input_pkl)` additionally checks the
serialized artifact content and identity. The exact relationship is:

```text
actual sumw2 process set == required_sumw2_processes
```

For the recovered category-limited Run-2 shape, the expected interpretation is:

```text
source_application_regions:
  isAR_1l
  isSR_1l
  isSR_2lOS

applicable_products:
  nonprompt: true
  flips: false

generated_nonprompt_processes:
  nonpromptUL16
  nonpromptUL16APV
  nonpromptUL17
  nonpromptUL18

generated_flips_processes: []
```

The example describes the recovered artifact shape, not a universal result for
every family. Applicability remains family-specific and follows the
authoritative application-axis evidence.
