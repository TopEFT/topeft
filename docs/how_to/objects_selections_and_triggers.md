# Change objects, selections, or triggers

Use this guide for an analysis-policy change to object roles, working points,
trigger lists, event filters, object cleaning, or dataset/sample overlap. Read
the [object and event-selection reference](../reference/objects_selections_and_triggers.md)
first. A reusable predicate or generic dataset-overlap algorithm belongs in
`topcoffea`, not in this procedure.

## Start from the canonical owners

- Analysis thresholds and role parameters:
  [`topeft/params/params.json`](../../topeft/params/params.json)
- Analysis predicates:
  [`topeft/modules/object_selection.py`](../../topeft/modules/object_selection.py)
- Triggers, filters, and dataset priority:
  [`topeft/modules/event_selection.py`](../../topeft/modules/event_selection.py)
- Processor attachment and selection order:
  [`analysis_processor.py`](../../analysis/topeft_run2/analysis_processor.py)
- Shared luminosity and b-tag working-point values:
  [`topcoffea/params/params.json`](https://github.com/TopEFT/topcoffea/blob/HEAD/topcoffea/params/params.json)

Do not copy a threshold or working-point table into a second config. Change its
owner and keep the caller selecting that owner.

## Select an existing object policy

Run-3 MVA-based lepton selection is enabled by the direct CLI. The diagnostic
comparison path disables it explicitly:

```bash
python analysis/topeft_run2/run_analysis.py <sample.json> \
  --noRun3MVA --hist-list njets
```

This changes object selection and all dependent categories and observables. It
does not merely change execution. Use the repository wrapper and environment
procedure in [production](production.md) before running the command; the
snippet shows the physics-bearing option only.

## Change an object threshold or role

1. Locate the current value in `params.json` and the predicate that consumes
   it. If the predicate takes a caller parameter, keep the value at the caller
   rather than hard-coding a second default.
2. Identify every era, flavor, and analysis mode that reaches the branch.
3. Follow the selected collection into scale factors, triggers, categories,
   and observables. A loose, fakeable, and tight role change has different
   data-driven consequences.
4. Update the owning value/predicate and the reference description together.
5. Run the focused object/event and processor tests, including
   `tests/test_event_selection_lepton_tau.py` and any era-specific correction
   test reached by the changed role.

The repository establishes the current value and effect. Record no motivation
unless a separate scientific authority supports it.

## Change a b-tag working point

1. Decide whether the change is to the shared numeric table or to the `topeft`
   choice of tagger/working-point name.
2. For a shared-table change, follow the
   [`topcoffea` extension guide](https://github.com/TopEFT/topcoffea/blob/HEAD/docs/physics_extension_guides.md)
   and validate its correction schema.
3. For an analysis-policy change, update the `topeft` selector/caller and trace
   both the tag collection and Method1a event weight.
4. Check category migration and central/up/down weight behavior together.

A threshold-only change that ignores its b-tag scale-factor consumer is
incomplete.

## Change triggers, filters, or dataset priority

1. Edit the era/channel registry in `topeft/modules/event_selection.py`.
2. Keep trigger lists separate from data primary-dataset precedence. The
   shared overlap helper consumes the precedence; it does not choose it.
3. Check data and MC behavior separately. Dataset exclusivity applies to data,
   while trigger masks can apply to both.
4. Confirm that the mask is combined before category filling for nominal and
   applicable object variations.
5. Exercise the relevant preflight/event-selection tests and a focused
   category-mask test.

## Change cleaning or overlap policy

Name the kind of overlap first:

- reconstructed-object cleaning belongs with object policy;
- primary-dataset overlap uses the shared algorithm with `topeft` precedence;
- ttgamma or other sample-family overlap belongs with sample roles.

Update only the owner of that policy and then inspect all affected
multiplicities, source roles, and categories. For ttgamma, continue with
[sample roles and normalization](sample_roles_and_normalization.md).

## Documentation and validation closure

Update the object reference and any affected category/correction page. Focused
validation should cover the selector, the processor consumer, and at least one
downstream category or weight consequence. A passing helper unit test alone is
not enough when the analysis call site changed.
