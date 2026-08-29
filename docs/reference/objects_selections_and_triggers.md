# Objects, selections, and triggers

This page specifies the maintained physics-facing object and event-selection
layer consumed by `AnalysisProcessor`. Concrete analysis cuts, working-point
choices, trigger maps, and dataset policy belong to `topeft`. Reusable
predicates and generic overlap utilities belong to `topcoffea`; see the
[shared-interface index](shared_topcoffea_interfaces.md).

## Object roles

Leptons are assigned analysis roles rather than represented by one universal
selection. Preselected or loose objects support event topology and
data-driven inputs; fakeable objects support nonprompt estimation; tight
objects define the prompt signal-like selection. Run-2 and Run-3 branches use
their era-appropriate inputs, and Run-3 can use the maintained MVA-based path.
Disabling that path is therefore a physics-selection choice, not an execution
setting.

Tau selections are active only for tau-enabled analysis modes. Jet collections
are corrected before tight-jet, forward-jet, and b-tag roles are derived. The
selected tagger and working point are analysis policy; shared numeric working-
point values selected through `get_tc_param` are owned by
`topcoffea/params/params.json`.

The implementation authorities are `topeft/modules/object_selection.py`, the
processor call sites, `topeft/params/params.json`, and the selected shared
predicate/working-point interfaces.

## Cleaning and overlap are separate contracts

Object cleaning removes geometrical or reconstruction overlap between
collections before multiplicities and observables are derived. Trigger-dataset
overlap removal instead prevents a data event accepted by more than one
primary-dataset path from being counted twice. Sample-family masks, including
the ttgamma policies, partition simulated source roles. These operations have
different inputs and must not be collapsed into one generic “overlap” switch.

`topcoffea.modules.event_selection` supplies reusable overlap and SFOS/Z-window
mechanisms. `topeft/modules/event_selection.py` owns concrete era trigger
lists, data-dataset ordering, filters, channel masks, and region thresholds.

## Trigger and event-cleaning semantics

Trigger masks depend on era, channel, and—for data—the primary dataset. Event
filters apply the maintained cleaning requirements before category filling.
The masks act on the nominal accepted event population and are consistently
combined with applicable varied-object selections. Dataset exclusivity is a
data policy; MC trigger application does not require primary-dataset
deduplication.

The processor consumes these masks before constructing CR/SR categories. A
trigger or filter change can therefore affect accepted yields in every
downstream observable and systematic view.

## Era and channel applicability

- Run-2 and Run-3 object branches use different maintained inputs where the
  source dispatch says so.
- Tau definitions affect tau-enabled channels only.
- Forward-jet policy affects forward/all-analysis categories and interacts with
  the corrected-jet/JER policy.
- Flavor, charge, Z-window, and multiplicity helpers apply only to compatible
  lepton channels.
- Dataset-overlap policy applies to data and depends on the selected primary
  dataset; sample-role overlap policy applies to the named simulated families.

## Variation behavior and downstream effects

An object shift can alter whether an object passes its role, so the processor
reevaluates dependent selections, category masks, and observables. Weight-only
variations do not redefine the objects. This difference is part of the
systematic contract, not merely an implementation detail.

Selected collections feed object multiplicities, lepton flavor and charge
structure, jet and b-tag counts, tau and forward categories, and all
object-derived observables. See
[categories and observables](categories_and_observables.md) and
[corrections, weights, and systematics](corrections_weights_and_systematics.md).

## Known contract boundaries

Registered, commented, or placeholder definitions are not promoted to current
analysis policy without a maintained consumer. The audit found no repository
authority for the scientific motivation of several thresholds and working
points. Their current values and effects are documentable, but this page does
not claim why they were chosen.

## Defaults, example, and modification route

Concrete analysis defaults live in
[`params.json`](../../topeft/params/params.json),
[`object_selection.py`](../../topeft/modules/object_selection.py), and
[`event_selection.py`](../../topeft/modules/event_selection.py). The shared
b-tag working-point values live in
[`topcoffea/params/params.json`](https://github.com/TopEFT/topcoffea/blob/HEAD/topcoffea/params/params.json).

Run-3 MVA lepton selection is normally enabled; `--noRun3MVA` is the explicit
physics-control example that disables that branch and changes all dependent
categories/observables. Threshold, WP, trigger, filter, cleaning, and overlap
changes are grouped under
[change objects, selections, or triggers](../how_to/objects_selections_and_triggers.md).
