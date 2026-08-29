# Change categories or observables

Use this guide to add or change a maintained category group, observable,
processing axis, or wrapper histogram matrix. Read
[categories and observables](../reference/categories_and_observables.md) and
[flexible binning](flexible_binning.md) first.

## Canonical owners

- Main category registry:
  [`topeft/channels/ch_lst.json`](../../topeft/channels/ch_lst.json)
- Specialist diboson registry:
  [`topeft/channels/ch_lst_diboson.json`](../../topeft/channels/ch_lst_diboson.json)
- Processing axes and observable metadata:
  [`topeft/modules/axes.py`](../../topeft/modules/axes.py)
- Processing/fitting view resolver:
  [`topeft/modules/axis_binning.py`](../../topeft/modules/axis_binning.py)
- Processor category masks and value construction:
  [`analysis_processor.py`](../../analysis/topeft_run2/analysis_processor.py)
- Production category/histogram matrices:
  [`run_cr.sh`](../../analysis/topeft_run2/run_cr.sh)

`ch_lst_tmp.json` is not a maintained target. Do not copy a category there and
call it current policy.

## Select an existing category and observable

`2los_CRZ` is a maintained category group in `ch_lst.json`, and `invmass` is a
maintained observable in `axes.py`. A focused direct selection is:

```bash
python analysis/topeft_run2/run_analysis.py <sample.json> \
  --skip-sr \
  --category-groups 2los_CRZ \
  --hist-list invmass
```

This example illustrates the chain: the CLI selects the registered group, the
processor builds its event mask, and the observable registry supplies the
filled axis. It is not a production recommendation; use the production guide
for a governed run.

## Add or change a category group

1. Define the event mask at the processor/event-selection owner. Reuse shared
   SFOS/Z helpers without moving concrete mass windows into `topcoffea`.
2. Add the group and leaf labels to `ch_lst.json` using the existing schema.
3. Confirm SR/CR scope, analysis-mode compatibility, and category token
   parsing. A name alone is not implementation evidence.
4. Add the group to a maintained wrapper profile only if it belongs in that
   production matrix.
5. Check every object variation that can migrate events across the new mask.
6. Validate `tests/test_run_analysis_category_groups.py`, processor preflight,
   and affected plot/card channel validation.

The mask and registry establish current behavior. Record no region motivation
unless a separate authority exists.

## Add an observable

1. Add the processing-axis definition and metadata in `axes.py`.
2. Construct the value in the maintained processor branch and make its
   channel/analysis applicability explicit.
3. Add the observable to the allowed histogram groups or wrapper matrix only
   where it is meant to be produced.
4. Decide whether fitting needs the processing view or an exact aggregated
   view. Add fitting policy in `axis_binning.py`, not in a plot/card caller.
5. Validate nominal, object-shift, weight-shift, EFT, and sumw2 alignment as
   applicable.
6. Update plotting/card configuration only after the produced artifact
   authority exists.

## Change binning

Use [flexible binning](flexible_binning.md) for the exact schema procedure.
Processing-edge changes are upstream and can invalidate every downstream
view. Fitting-default or exact-channel changes are downstream views and must
remain exact aggregations of processing bins.

## Keep specialist categories separate

Diboson categories use `ch_lst_diboson.json` and the specialist processor/CLI.
Make the same registry, observable, and validation checks there, but do not add
specialist groups to the main processor solely to make a wrapper convenient.
See [specialist interfaces](../reference/specialist_interfaces.md).

## Documentation and test closure

Link the new or changed reference entry directly back to this guide. Validate
category selection, observable construction, binning resolution, wrapper
coverage, plotting channel resolution, and datacard late rebinning as affected.
