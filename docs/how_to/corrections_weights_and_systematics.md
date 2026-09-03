# Change corrections, weights, or systematics

Use this guide for analysis-owned payload selection, era dispatch, working
points, weight attachment, variation exposure, and forward-JER policy. Shared
evaluators, factories, and packaged shared payloads are owned by `topcoffea`.
Read [corrections, weights, and systematics](../reference/corrections_weights_and_systematics.md)
before changing either side.

## Start from the maintained authorities

- Analysis dispatch and rate evaluators:
  [`topeft/modules/corrections.py`](../../topeft/modules/corrections.py)
- Analysis parameters and policy sets:
  [`topeft/params/params.json`](../../topeft/params/params.json)
- Fake-rate payload family:
  [`topeft/data/fakerates`](../../topeft/data/fakerates)
- Charge-flip payload family:
  [`topeft/data/fliprates`](../../topeft/data/fliprates)
- Trigger scale-factor payload family:
  [`topeft/data/triggerSF`](../../topeft/data/triggerSF)
- Rate-systematic files:
  [`rate_systs_run2.json`](../../topeft/params/rate_systs_run2.json) and
  [`rate_systs_run3.json`](../../topeft/params/rate_systs_run3.json)
- Processor correction and weight consumers:
  [`analysis_processor.py`](../../analysis/topeft_run2/analysis_processor.py)

Shared correction and payload authorities are indexed in the
[cross-repository ownership page](../reference/shared_topcoffea_interfaces.md).

## Select maintained variations

The direct CLI produces nominal content unless systematic processing is
requested. This representative physics-only fragment enables variations while
keeping the default forward policies:

```bash
python analysis/topeft_run2/run_analysis.py <sample.json> \
  --do-systs --hist-list njets met
```

The default forward policy suppresses stochastic JER in the maintained eta
band and resolves eta-band pT application with `auto`. To study the supported
alternatives, make each deviation explicit:

```bash
python analysis/topeft_run2/run_analysis.py <sample.json> \
  --do-systs \
  --no-suppress-forward-eta-stochastic-jer \
  --fwd-eta-band-pt-apply off \
  --hist-list njets met
```

These switches change calibrated-object and systematic behavior. They are not
executor settings. No scientific preference among the policies is asserted by
the example.

## Add or update an analysis-owned payload

1. Identify the existing selector, era mapping, payload schema, and consuming
   object or weight.
2. Add the payload in its existing family; do not introduce a user-local path
   or copy a shared `topcoffea` payload into `topeft`.
3. Update the selector only for the intended eras/samples and fail closed for
   unsupported inputs.
4. Preserve central and named variation fields expected by the consumer.
5. Trace the result through object selection or weight assembly and the
   histogram systematic label.
6. Run the focused evaluator, era-dispatch, processor, and downstream tests.

Changing a numeric payload is a physics-policy action and requires its own
scientific approval; this documentation describes the software route only.

## Add an era to an existing correction

Update the smallest existing era map. Confirm the input collection and payload
schema, data/MC applicability, correction order, central behavior, and exposed
variations. Then check every consumer reached by that era. Relevant focused
tests include `tests/test_muon_momentum_correction_order.py`,
`tests/test_tau_energy_corrections.py`,
`tests/test_tau_correction_wp_policy.py`, and
`tests/test_trigger_syst_naming.py`, depending on the mechanism.

## Expose a variation

1. Verify that the owning evaluator or payload already defines the variation.
2. Add its name at the correction/weight registry, not only at histogram fill.
3. Decide whether it is an object shift or a weight modifier.
4. For an object shift, preserve corrected collection, selection, category,
   observable, and dependent-weight propagation.
5. For a weight modifier, preserve nominal object/category semantics.
6. Validate nominal/up/down completeness and downstream card/plot consumers.

Do not use `--do-renormfact-envelope`; it is deprecated and exits before
analysis. Scale-weight extraction exists in `topcoffea`, but envelope and
template policy must have an active caller-owned contract.

## Change a shared mechanism

If the required change is to a correctionlib boundary, JEC stack, corrected-
jet/Type-1 MET mechanism, generic weight helper, or packaged shared payload,
stop editing `topeft` and use the
[`topcoffea` physics extension guide](https://github.com/TopEFT/topcoffea/blob/HEAD/docs/physics_extension_guides.md).
After that change is validated, return here to select it explicitly for the
analysis and update both repositories' ownership links.

## Validation closure

Validate the mechanism or payload schema, the `topeft` dispatch, processor
propagation, and at least one downstream plot/card or category effect. Update
the reference page with the concrete authority and applicability. Do not copy
payload numbers into prose.

## TAU change checklist

For a tau change, update the owning selector or helper rather than payload
registration alone. Keep the analysis jet-to-tau fake families
`lepSF_taus_fake_run2` and `lepSF_taus_fake_run3` separate from genuine-tau
POG `VSjet`, `VSe`, and `VSmu` weights and from source/decay-mode energy
shifts. Preserve the standard versus taufitter and data/MC gates, then validate
the exact systematic naming and processor routing. A packaged payload does not
activate itself in the analysis. See the [TAU family reference](../reference/corrections_weights_and_systematics.md)
for the bounded interface contract.
