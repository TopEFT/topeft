# Change sample roles or normalization inputs

Use this guide to add a maintained sample, change an analysis sample role, or
update the metadata path through which normalization is consumed. Read
[sample roles and normalization](../reference/sample_roles_and_normalization.md)
before editing.

## Find the real owner

- Active sample cfg/JSON selection and validation:
  [`analysis/topeft_run2/run_analysis.py`](../../analysis/topeft_run2/run_analysis.py)
  and
  [`topeft/modules/production_sample_profile.py`](../../topeft/modules/production_sample_profile.py)
- Analysis role sets:
  [`topeft/params/params.json`](../../topeft/params/params.json)
- ttgamma classification:
  [`topeft/modules/ttgamma_photon_history.py`](../../topeft/modules/ttgamma_photon_history.py)
- Sum-of-weights runner and processor:
  [`run_sow.py`](../../analysis/topeft_run2/run_sow.py) and
  [`sow_processor.py`](../../analysis/topeft_run2/sow_processor.py)
- Shared luminosity defaults:
  [`topcoffea/params/params.json`](https://github.com/TopEFT/topcoffea/blob/HEAD/topcoffea/params/params.json)

Use the currently selected repository sample JSON/cfg as the schema example.
Do not substitute a produced user-local JSON or infer metadata from a filename.

## Add a sample to the maintained universe

1. Start from a current sample record for the same era and data/MC/EFT class.
2. Supply the required identity, era, files, data/MC flag, histogram-axis name,
   numeric cross section and generated-event sums where applicable, and EFT WC
   metadata only for a compatible source.
3. Add the record to the maintained cfg/JSON authority and update the active-
   universe profile rather than bypassing certification.
4. Check central/private exclusivity and required signal/data-driven
   contributors.
5. Run JSON metadata, active-universe, and processor-preflight tests before any
   production run.

## Change `lo_xsec_samples`

The canonical structure in `params.json` is a list of sample names:

```json
"lo_xsec_samples": [
  "TTGamma_centralUL16APV",
  "TTGamma_centralUL16"
]
```

The excerpt shows the structure, not the complete list. Add or remove a name
only when the intended rate/systematic role is scientifically approved. Do not
put a numeric cross section in this list. Numeric values remain in sample
metadata. After a membership change, validate both the main and specialist
processor branches that test the set and the affected datacard rate policy.

## Change ttgamma source treatment

Use the supported CLI policy for a focused diagnostic rather than editing
process labels:

```bash
python analysis/topeft_run2/run_analysis.py <sample.json> \
  --ttgamma-sample-role-policy run2_nlo_inclusive \
  --hist-list njets
```

The production default is `split`; the alternate is constrained to its
supported Run-2 diagnostic use. To extend the policy itself, update the
classifier, supported-policy registry, CLI resolver, processor masks, and
`tests/test_ttgamma_photon_history.py` together.

## Change data-driven source roles

Update the prompt, conversion, subtraction, or target set in its `params.py` or
product-policy owner. Then validate process membership, applicable era/region,
required source content, variation preservation, and sidecar certification.
Continue with [the nonprompt guide](nonprompt.md).

## Refresh normalization information

`run_sow.py` is the maintained specialist calculation path. Regenerating sums
is execution and is not performed by documentation changes. In a separately
authorized run, select the exact sample JSON, retain the output/provenance, and
update only the canonical metadata owner after validating completeness. Do not
teach the main processor to recompute missing normalization opportunistically.

## Closure

Validate sample schema, active-universe policy, role-specific masks,
normalization, EFT treatment, and downstream process grouping. Update the
reference page and the relevant production profile when the maintained
universe changes.
