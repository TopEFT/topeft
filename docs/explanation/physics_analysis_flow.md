# Physics analysis flow

The maintained processor is easiest to understand as a sequence of physics
decisions rather than as a list of Python functions. Sample metadata first
defines what an input represents. Corrections establish object kinematics
before the dependent object roles are selected. Event-weight components are
then accumulated as their required sample, object, and event inputs become
available. Selected objects and event masks define categories and observables.
Systematic variations repeat the affected part of that chain, and the
resulting histogram products feed the data-driven, plotting, datacard, and EFT
consumers.

This page explains how those decisions fit together. Exact interfaces and
configured values remain in the linked reference pages.

## 1. Interpret the sample before the event

The sample JSON or cfg supplies dataset identity, era, data/MC status,
normalization metadata, EFT metadata, and analysis-specific sample roles.
Those fields determine which correction and weight branches are meaningful,
whether EFT coefficients are prepared, and which overlap or data-driven masks
apply. A sample-role set is policy, not a replacement for numeric sample
metadata. In particular, `lo_xsec_samples` classifies samples for a rate and
systematic treatment; it is not a table of cross sections.

See [sample roles and normalization](../reference/sample_roles_and_normalization.md)
and [production configuration](../reference/production_configuration.md).

## 2. Correct objects and assemble weights

Object corrections alter the collections that later selections inspect. Event
weights instead alter a selected event's contribution. Jet-energy corrections
also propagate into MET, so a shifted jet collection can change both category
membership and MET observables. Lepton, tau, b-tag, pileup, trigger, generator,
and other scale factors contribute only where their era, sample, and object
requirements are satisfied.

`topcoffea` owns reusable factories, evaluators, and payload interfaces.
`topeft` owns the concrete era dispatch, working-point selection, enabled
variations, and processor call sites. See
[corrections, weights, and systematics](../reference/corrections_weights_and_systematics.md)
and the [shared-interface ownership index](../reference/shared_topcoffea_interfaces.md).

## 3. Define objects, accept events, and prevent overlap

Loose, fakeable, and tight leptons, taus, jets, forward jets, and b-tagged jets
are constructed before channel masks. Trigger and event-cleaning masks then
restrict the accepted events. Data primary-dataset overlap removal and
sample-family masks prevent the same physical event or source role from being
counted in more than one maintained sample category.

These are distinct operations: object cleaning removes overlap between
reconstructed objects, while dataset and sample-role policies remove overlap
between event sources. Their concrete analysis definitions are in
[objects, selections, and triggers](../reference/objects_selections_and_triggers.md).

## 4. Construct categories and observables

Named categories combine object multiplicities, charge and flavor structure,
Z-window masks, jet and b-tag multiplicities, tau content, and forward-object
requirements. Observables are computed for the selected objects and event
topology, then associated with processing axes. Downstream fitting may use a
coarser exact view of the same processing axis; it does not redefine the
observable.

The channel registry and observable/axis registries are implementation
authorities. Their presence establishes the current mask or binning, but does
not by itself establish why a collaboration chose that region or threshold.
See [categories and observables](../reference/categories_and_observables.md).

## 5. Propagate variations through the affected decisions

A weight variation reuses nominal objects and event masks while changing the
event contribution. An object variation can change kinematics, selections,
category membership, observables, and dependent weights. The processor keeps
the histogram-filling contract common across the nominal and applicable
variations so the systematic axis describes comparable physics products.

Systematic production is an explicit analysis policy. A registered name alone
does not prove that a variation is enabled or filled. See
[the processor execution map](../reference/analysis_processor.md) and
[corrections, weights, and systematics](../reference/corrections_weights_and_systematics.md).

## 6. Fill EFT-aware histograms and statistical companions

For ordinary samples, histogram content is filled as scalar weighted yields.
For EFT-capable samples selected for EFT treatment, the processor prepares the
coefficient content consumed by `HistEFT`. Sumw2 companions represent
statistical second moments and remain separate from physics nuisance
variations. Provenance records the resolved policy and artifact relationships
needed by downstream validation.

The generic EFT algebra and histogram mechanism belong to `topcoffea`; sample
treatment and analysis consumption belong to `topeft`. See
[HistEFT](../reference/histeft.md), [sumw2](../reference/sumw2.md), and
[histogram artifacts](../reference/histogram_artifacts.md).

## 7. Transform and consume the products

Certified source histograms may be transformed into nonprompt or charge-flip
products. Plotting selects physics-facing regions, groupings, and observable
views. Datacard construction maps categories and distributions to rates,
shapes, statistical terms, and EFT scaling records. Each consumer validates
the artifact contract it needs rather than inferring missing content.

See [data-driven estimation](../reference/data_driven_estimation.md),
[plotting](../reference/plotting.md), and
[datacards and scalings](../reference/datacards_and_scalings.md).

## Scientific-authority boundary

The checked-in processor, registries, parameters, payload selectors, and tests
establish the implemented behavior described above. Named external calibration
authorities are retained where the source identifies them, but their
derivations are not reopened here. Where no physics-rationale authority was
found, these pages describe the choice and its observable consequence without
claiming an optimization, historical motivation, or collaboration preference.

## Concrete path through the flow

The maintained `2los_CRZ` category and `invmass` observable make the sequence
concrete. The category group is registered in
[`ch_lst.json`](../../topeft/channels/ch_lst.json); the observable is registered
in [`axes.py`](../../topeft/modules/axes.py). For a selected era/sample, the
processor corrects objects, applies trigger/cleaning/overlap masks, constructs
the opposite-sign CRZ mask, computes dilepton invariant mass, and fills the
requested histogram. Enabling systematics adds the applicable object- and
weight-variation paths described above. The example shows the owners and
downstream propagation; it does not claim why the region or binning was chosen.
