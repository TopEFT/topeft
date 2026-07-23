# Diboson $N_{\text{jets}}$ scale factors (Run 3)

`diboson_sf_run3.py` derives diboson scale factors from the scalar `njets`
histogram. Its default final binning is `[0, 1, 2, 3, 4, 5, 6]`. Statistical
uncertainties are propagated by default from the paired scalar
`njets_sumw2` histogram.

## Process roles and configuration

Every selected process must appear exactly once in the user-facing YAML
configuration. Role entries are exact resolved labels; regular expressions,
prefixes, and substrings are not interpreted. The `data`, `background`, and
`diboson` roles must be nonempty. Processes assigned to `ignored` enter neither
the central estimator nor its variance.

The tracked [`diboson_sf_run3_config.yml`](diboson_sf_run3_config.yml) is a
complete configuration for the focused current-format fixture. For another
input, copy that file and replace every role list with the exact labels on that
input's `process` axis, then pass the file with `--config`. There is no hidden
Python fallback process list.

```yaml
diboson:
  propagate_statistical_uncertainties: true
  process_roles:
    data: [exact_data_process]
    background: [exact_prompt_background_process]
    diboson: [exact_diboson_process]
    ignored: []
```

Duplicate labels, overlap between roles, labels absent from the nominal
histogram, and selected but unclassified processes are errors.

## Statistical contract

For each requested year, channel, and final bin, the script first aggregates
the nominal role components and their second moments over the same source
`njets` bins:

```text
d = data nominal sum                 var_d = data njets_sumw2 sum
b = background nominal sum           var_b = background njets_sumw2 sum
v = diboson nominal sum              var_v = diboson njets_sumw2 sum
```

It then evaluates the signed central estimator and its independent absolute
variance:

```text
r = (d - b) / v
var_r = (var_d + var_b) / v**2 + ((d - b)**2 / v**4) * var_v
sigma_r = sqrt(var_r)
```

The script does not substitute Poisson counts, derive variance from nominal
contents, or compute source-bin ratios before rebinning. A cancelled numerator
keeps `r = 0` and can retain positive uncertainty. A negative numerator remains
negative. A nonfinite or nonpositive diboson denominator is a blocking error.

When propagation is enabled, both histogram keys are required and their axes,
categories, edges, coverage, and flow semantics must match. Second moments must
be finite and nonnegative. All validation and calculations finish before the
CLI writes any final JSON or plot.

## Enabled and disabled modes

The resolution order is explicit CLI, then YAML configuration, then the
default value `true`:

```text
--propagate-statistical-uncertainties
--no-propagate-statistical-uncertainties
```

Disabled mode uses `njets` only. It neither accesses nor validates
`njets_sumw2`; JSON statistical arrays are `null`, and the plot visibly states
`statistical uncertainties disabled`. The switch affects only this consumer
calculation and never changes processor allocation or the resolved sumw2
storage plan.

## Running the CLI

Use role configurations matching each selected input exactly. A single-input
invocation keeps the original one-config form:

```bash
python analysis/diboson_njets/diboson_sf_run3.py \
  --pkl input_2022.pkl.gz \
  --config roles_2022.yml \
  --channel 3l_CR \
  --year 2022 \
  --output-dir output
```

A shared multi-year input reuses one exhaustive config containing the full role
set for every selected period:

```bash
python analysis/diboson_njets/diboson_sf_run3.py \
  --pkl combined.pkl.gz \
  --config roles_combined.yml \
  --channel 3l_CR \
  --year 2022 2022EE \
  --output-dir output
```

The same shared form supports automatic discovery and writes each discovered
period plus the combined `all` result:

```bash
python analysis/diboson_njets/diboson_sf_run3.py \
  --pkl combined.pkl.gz \
  --config roles_combined.yml \
  --channel 3l_CR \
  --year all \
  --output-dir output
```

Independent year files require matching configs in the same positional order:

```bash
python analysis/diboson_njets/diboson_sf_run3.py \
  --pkl input_2022.pkl.gz input_2022EE.pkl.gz \
  --config roles_2022.yml roles_2022EE.yml \
  --channel 3l_CR \
  --year 2022 2022EE \
  --output-dir output
```

Input and config templates may instead use the literal `{year}` placeholder:

```bash
python analysis/diboson_njets/diboson_sf_run3.py \
  --pkl 'input_{year}.pkl.gz' \
  --config 'roles_{year}.yml' \
  --channel 3l_CR \
  --year 2022 2022EE \
  --output-dir output
```

An input template may also be paired with an explicit config list, or an
explicit input list with one config template. Configs always contain exact
resolved labels: one config can be reused only for a shared input containing
its full role set, while independent files require one matching config per
input. The CLI propagation flags apply uniformly to the whole invocation. When
there is no CLI override, every assigned config must resolve the same
`propagate_statistical_uncertainties` state. A `{year}` config template is not
valid with `--year all`, because discovery requires one shared exhaustive
config.

## Outputs and provenance

Each year directory contains:

- `diboson_sf_{year}.json`, preserving the central scale-factor mapping and
  adding aligned variance, uncertainty, formula, configuration-source, role,
  input-identity, bin-membership, and validation provenance;
- `diboson_sf_{year}_linear_fit.json`, preserving the unweighted central linear
  fit definition;
- `diboson_sf_{year}.png`, using the propagated uncertainties as point error
  bars when enabled or the visible disabled annotation otherwise.

The fixture at `tests/data/run3_histogram.pkl.gz.base64` is a small deterministic
pickle generated with the pinned current environment. It includes multiple
processes in several roles, weighted data with `sumw2 != nominal`, ignored
content, and two source bins per final bin.
