# Corrections, weights, and systematics

Corrections establish calibrated object kinematics or multiplicative event
weights before histogram filling. Systematic variations expose the configured
alternatives to those quantities. This page owns the concrete `topeft`
dispatch and consumption policy; reusable algorithms and shared payload
contracts belong to `topcoffea`.

## Ownership model

| Responsibility | Owner |
| --- | --- |
| Era/sample dispatch, selected tags and payload families, working points, enabled variations, and processor application | `topeft` |
| Correction evaluators, jet/MET factories, reusable weight extraction, generic selection helpers, and packaged shared payload interfaces | `topcoffea` |
| Analysis-owned fake, flip, trigger, rate-systematic, and missing-parton payloads | Their checked-in `topeft` parameter/data owners |
| Calibration derivation | The named external authority recorded with the payload or source; not re-derived here |

The [shared-interface index](shared_topcoffea_interfaces.md) links each generic
mechanism to its owner.

## Object corrections

Muon and electron momentum handling, jet-energy corrections and resolution,
MET propagation, veto maps, and related object corrections run before the
affected selection. The current era, data/MC status, collection name, and
selected payload determine dispatch. Corrected jets feed both jet categories
and corrected MET, so a jet shift can migrate an event through categories and
change MET observables.

Forward stochastic-JER suppression is a split contract. `topcoffea` supplies
the factory hook; `topeft` resolves and enables the analysis policy. The
forward eta-band pT policy is likewise a concrete analysis choice. The source
establishes their behavior and interaction, but the audit found no independent
physics-rationale authority for the choices.

## Event weights

The nominal MC weight is assembled only from components applicable to the
sample, era, and selected objects. Components include normalization, pileup,
lepton and trigger scale factors, b-tag factors, prefiring where applicable,
and generator-derived parton-shower and scale terms. Data do not receive MC
normalization or generator weights. Fake and charge-flip quantities participate
through the data-driven contracts rather than becoming universal prompt-MC
scale factors.

`lo_xsec_samples` is a sample-role set used when resolving a rate/systematic
branch. It is not the source of numeric cross sections; those remain in sample
metadata. See [sample roles and normalization](sample_roles_and_normalization.md).

## Central and varied behavior

The nominal path uses the resolved central object collections and central
weight components. When systematic production is enabled, two propagation
patterns are maintained:

- object variations rebuild the affected collection and reevaluate dependent
  selections, categories, observables, and weights;
- weight variations retain nominal objects and masks while changing the event
  contribution.

Variation names and applicability are controlled by the correction helpers,
payload schemas, and processor registries. A variation is not active merely
because a payload contains a similarly named field. The deprecated
renormalization/factorization-envelope CLI entry has no active consumer and is
not an alternative to the maintained scale-weight flow.

## Applicability and failure boundaries

Era, data/MC status, sample family, required branches, collection type, and
analysis mode all constrain applicability. Missing required payloads, mappings,
branches, or incompatible correction schemas are failures at their owning
interface; documentation must not imply a nominal-only fallback unless the
source defines one.

The resulting central and varied objects or weights affect selected yields,
category migration, observable shapes, histogram systematic labels, plotting,
and datacard nuisances. Statistical sumw2 companions are separate from these
physics variations. See [sumw2](sumw2.md) and
[datacards and scalings](datacards_and_scalings.md).

## Scientific-authority boundary

Checked-in dispatch, selectors, payload paths, and tests establish which
correction is applied and what changes downstream. Named POG or calibration
authorities may establish payload provenance. Neither source use nor a payload
name alone establishes why the analysis chose a family, working point, or
uncertainty subset, so this page does not supply that missing motivation.

## Defaults, variation example, and modification route

Analysis-owned authorities are
[`corrections.py`](../../topeft/modules/corrections.py),
[`params.json`](../../topeft/params/params.json), the
[`fakerates`](../../topeft/data/fakerates),
[`fliprates`](../../topeft/data/fliprates), and
[`triggerSF`](../../topeft/data/triggerSF) payload families, and the current
[`rate-systematic files`](../../topeft/params/rate_systs_run3.json).
Shared payload/mechanism authorities are linked from the
[ownership index](shared_topcoffea_interfaces.md).

Nominal processing is the direct CLI baseline. `--do-systs` adds applicable
object and weight variations. The forward policy is represented by the default
stochastic-JER suppression plus `--fwd-eta-band-pt-apply auto`; explicit
disable/`off` controls show a supported alternative without assigning a
scientific preference. See
[change corrections, weights, or systematics](../how_to/corrections_weights_and_systematics.md)
for era, payload, variation, and shared-mechanism routes.
