# Categories and observables

Categories define which selected events occupy each analysis region.
Observables define the physical quantities filled for those events. Processing
binning records the information produced by the processor; fitting binning is
an exact downstream view used by statistical consumers.

## Category authority

`topeft/channels/ch_lst.json` is the maintained main category registry.
`AnalysisProcessor` combines its named groups with implemented masks for
lepton/tau multiplicity, charge, flavor, Z-window status, jet and b-tag
multiplicity, and forward-object content. Analysis-mode and wrapper controls
select which registered groups are active.

`ch_lst_tmp.json` is not a maintained policy authority. The diboson registry is
a maintained specialist contract and remains separate from the core registry;
see [specialist interfaces](specialist_interfaces.md).

## CR and SR semantics

Control- and signal-region names refer to concrete combinations of the masks
implemented in the processor and channel registry. Those sources establish
event membership and downstream effects. They do not, by themselves,
establish the scientific motivation or optimization history of the regions.
This documentation therefore describes what each maintained mask selects
without inventing why it was chosen.

Trigger, cleaning, overlap, and object-role inputs are defined in
[objects, selections, and triggers](objects_selections_and_triggers.md).

## Observable authority

`topeft/modules/axes.py` owns the processing-axis definitions used by the
processor, while the processor owns channel-specific value construction and
fill eligibility. Observable names may represent lepton, jet, MET, invariant
mass, angular, or channel-specific quantities. A declared axis without a
maintained fill path is not current output; commented or unfilled legacy
generator/flavor diagnostics remain outside this contract.

Requested histogram groups constrain what is built and filled. They do not
alter the underlying definition of an observable.

## Processing and fitting views

`topeft/modules/axis_binning.py` resolves maintained processing and fitting
views. Processing edges must retain enough information for each supported
downstream exact aggregation. Fitting views may be family defaults or exact
channel-name overrides. Datacards and plots select a view; they do not silently
change the processor's physical quantity.

See [flexible binning](flexible_binning.md) for schemas and resolution rules.

## Systematic behavior

Object variations can change both the observable value and the category that
receives the event. Weight variations retain the nominal observable/category
view while changing its contribution. Every enabled variation is filled
through the common process/category/systematic contract, subject to the
variation's applicability.

Categories and observables feed histogram artifacts, data-driven products,
plots, card distributions, and EFT scaling records. The corresponding
consumer contracts are linked from
[the physics analysis flow](../explanation/physics_analysis_flow.md).

## Rationale boundary

The processor, channel registry, axis registry, and binning resolver establish
the current implementation. Where no separate physics-rationale authority was
found, this page does not claim why a region, observable, or bin edge was
chosen.

## Concrete anchor and modification route

The canonical registries are
[`ch_lst.json`](../../topeft/channels/ch_lst.json),
[`axes.py`](../../topeft/modules/axes.py), and
[`axis_binning.py`](../../topeft/modules/axis_binning.py). `2los_CRZ` paired
with `invmass` is a representative maintained category-to-observable path: the
registry selects the group, the processor applies its mask, and the axis owner
defines the filled quantity. It is not a universal category or observable
default.

Use [change categories or observables](../how_to/categories_and_observables.md)
for categories, axes, wrapper matrices, and specialist registries; use
[flexible binning](../how_to/flexible_binning.md) for processing/fitting views.

## Accepted fitting-topology anchor

The fitting authority is 129 SR category-distribution combinations and 555
fitting bins per run. The separate 1677-row Run 3 processing inventory is
descriptive processing evidence; it is not a fitting-bin count or a
statistical-adequacy authority.
