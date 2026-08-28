# Extend EFT inputs or consumers

Use this guide for `topeft` sample treatment, coefficient selection, processor
filling, SM-point consumption, or EFT-aware downstream validation. Reusable
coefficient algebra and `HistEFT` changes belong to `topcoffea`.

## Authorities

- Analysis treatment and filling:
  [`analysis_processor.py`](../../analysis/topeft_run2/analysis_processor.py)
- Direct coefficient selection:
  [`run_analysis.py`](../../analysis/topeft_run2/run_analysis.py)
- Current API/consumer contract: [HistEFT reference](../reference/histeft.md)
- Sample metadata and roles:
  [sample roles and normalization](../reference/sample_roles_and_normalization.md)
- Shared mechanism:
  [`topcoffea` EFT interfaces](https://github.com/TopEFT/topcoffea/blob/HEAD/docs/eft_interfaces.md)

There is no universal WC default. The selected sample metadata and requested
coefficient list are the concrete authority.

## Select a coefficient subset

`ctG` is used by maintained tests and examples. A focused direct selection is:

```bash
python analysis/topeft_run2/run_analysis.py <sample.json> \
  --wc-list ctG \
  --hist-list njets
```

The sample still decides whether EFT treatment is valid. Passing `--wc-list`
does not turn an ordinary sample into an EFT sample.

## Add EFT metadata for a new sample

1. Start from a current EFT-capable sample record and preserve its metadata
   schema.
2. Supply unique, nonempty WC names in the native coefficient order exposed by
   the source sample.
3. Classify the sample under the maintained EFT/SM/ignored treatment policy.
4. Check global requested-WC ordering and missing/duplicate-name failures.
5. Validate processor fill, SM evaluation, sumw2-at-SM, and serialization.

Relevant focused coverage includes `tests/test_sm_only_eft_treatment.py`,
`tests/test_eft_dataset_key_integration.py`,
`tests/test_analysis_processor_eft_sumw2.py`, and
`tests/test_histeft_api_contract.py`.

## Extend an analysis consumer

For a new data-driven, plot, card, or scaling consumer:

1. Decide whether it needs coefficient-bearing content or an evaluated
   ordinary-histogram view.
2. Make the evaluation point explicit. An empty mapping is the SM point; do
   not hide a nonzero benchmark as a helper default.
3. Preserve category, systematic, variance, and flow-bin contracts.
4. Reject incompatible ordinary/EFT input rather than guessing a projection.
5. Add focused parity coverage with an existing `HistEFT` consumer.

## Change shared EFT algebra or `HistEFT`

Use the
[`topcoffea` physics extension guide](https://github.com/TopEFT/topcoffea/blob/HEAD/docs/physics_extension_guides.md)
for coefficient helpers, storage, evaluation, or SparseHist substrate changes.
After shared validation, update the `topeft` consumer and reciprocal ownership
links. Do not promote legacy `WCPoint` or `WCFit` as the current analysis owner.

## Closure

Validate ordinary, EFT, and ignored sample treatment; coefficient order;
nominal/systematic fill; SM-point data-driven use; sumw2; pickle readback; and
the affected downstream consumer. Update the reference entry with a direct
link back to this procedure.
