# Shared topcoffea interfaces and responsibilities

`topeft` consumes reusable mechanisms from `topcoffea` while retaining its own
analysis policy. This index tells readers which repository provides each part. It
does not duplicate algorithms, equations, configured values, or extension
procedures.

## Responsibility split

- Read `topcoffea` documentation for reusable input/output, algorithm,
  variation, schema, and failure contracts.
- Read `topeft` documentation for the concrete era, sample, working point,
  payload, activation, category, and downstream analysis choice.
- When both participate, a policy change and a mechanism change are separate
  changes and require validation at both sides of the boundary.

## Corrections and calibrated objects

`topcoffea` provides the b-tag, pileup,
generator-weight helpers, JEC stack, corrected jets, Type-1 MET, Run-3 muon
correction, and packaged shared payload interfaces documented in
`topcoffea/docs/correction_interfaces.md`.
`topeft` configures the selected era/tag/payload, working point, enabled
variation, forward-JER policy, and processor application, as documented in
[corrections, weights, and systematics](corrections_weights_and_systematics.md).
The shared `params.json` and
packaged correction data provide the numeric values and payloads;
the selected analysis use is documented here in `topeft`.

The legacy corrected-MET factory is not the current Type-1 MET implementation used by
`topeft`; only a consumer that actually selects it should link it.

## Object and event selection

`topcoffea.modules.object_selection` supplies reusable jet predicates and
Run-3 jet-ID interpretation. `topcoffea.modules.event_selection` supplies
generic dataset-overlap and SFOS/Z-window mechanisms. Concrete lepton/tau/jet
cuts, trigger lists, dataset precedence, mass windows, filters, object-cleaning
policy, sample-role overlap masks, and category use belong to
[objects, selections, and triggers](objects_selections_and_triggers.md).

## EFT and histogram mechanisms

`topcoffea` implements coefficient algebra, helper transformations, and the
`HistEFT`/`SparseHist` mechanism documented in
`topcoffea/docs/eft_interfaces.md`. `topeft` configures per-sample treatment,
processor filling, SM-point consumption, and scaling/artifact use, as documented
in [HistEFT in `topeft`](histeft.md). `SparseHist` is generic software
infrastructure; it is not physics-bearing merely because `HistEFT` uses it.

## Change boundary

Analysis-policy changes route to the contextual `topeft` how-to guides.
Reusable mechanism, payload-packaging, factory, or EFT-algebra changes route to
[`topcoffea` physics extension guides](https://github.com/TopEFT/topcoffea/blob/run3_test_mmerged/docs/physics_extension_guides.md).
The shared references are
[`topcoffea` correction interfaces](https://github.com/TopEFT/topcoffea/blob/run3_test_mmerged/docs/correction_interfaces.md)
and
[`topcoffea` EFT interfaces](https://github.com/TopEFT/topcoffea/blob/run3_test_mmerged/docs/eft_interfaces.md).

Analysis changes use the contextual guides for
[objects and selections](../how_to/objects_selections_and_triggers.md),
[corrections and systematics](../how_to/corrections_weights_and_systematics.md),
[sample roles](../how_to/sample_roles_and_normalization.md), and
[EFT consumers](../how_to/eft.md). Low-level helper entries resolve to these
owning tasks instead of teaching direct edits to internal schema functions.
