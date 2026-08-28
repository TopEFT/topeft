# topeft documentation

This is the canonical map for the maintained `topeft` documentation. Current
analysis documentation describes TOP-26-006. Material retained to reproduce or
understand TOP-22-006 is labeled historical and is not part of the current
workflow.

The documentation is organized by the kind of help a reader needs:

- [Tutorials](#tutorials) teach a workflow in a guided sequence.
- [How-to guides](#how-to-guides) give task-oriented operating and extension
  procedures.
- [Reference](#reference) records exact commands, interfaces, configuration,
  schemas, defaults, and failure boundaries.
- [Explanation](#explanation) describes responsibilities, design choices, and
  boundaries between actors and artifacts.

If you are new to the repository, begin with the
[analysis workflow tutorial](tutorials/analysis_workflow.md). It introduces the
main actors before following a histogram artifact through plotting, datacard
creation, scaling finalization, and the EFTFit/Combine handoff.

## Tutorials

- [Analysis workflow](tutorials/analysis_workflow.md): the current TOP-26-006
  path from sample selection to the statistical-analysis handoff.
- [Histogram artifacts and HistEFT](tutorials/histogram_artifacts.md): how
  processor output is represented, inspected, and carried between workflow
  stages.

Older learning material is listed under [Historical material](#historical-material)
and is not a substitute for these current tutorials.

## How-to guides

### Production and artifacts

- [Run or extend the production entrypoints](how_to/production.md)
- [Change sample roles or normalization inputs](how_to/sample_roles_and_normalization.md)
- [Produce and recover data-driven nonprompt artifacts](how_to/nonprompt.md)
- [Run with Work Queue](how_to/work_queue.md)

### Objects, corrections, and categories

- [Change objects, selections, or triggers](how_to/objects_selections_and_triggers.md)
- [Change corrections, weights, or systematics](how_to/corrections_weights_and_systematics.md)
- [Change categories or observables](how_to/categories_and_observables.md)
- [Extend EFT inputs or consumers](how_to/eft.md)

### Plotting, cards, and statistical inputs

- [Make and configure analysis plots](how_to/plotting.md)
- [Create cards and finalize scalings](how_to/datacards_and_scalings.md)
- [Maintain missing-parton payloads](how_to/missing_parton_payloads.md)

### Histogram policy and binning

- [Select or extend sumw2 storage](how_to/sumw2.md)
- [Inspect or change processing and fitting binning](how_to/flexible_binning.md)

### Development and specialist tasks

- [Run focused tests](how_to/testing.md)
- [Extract fake-tau scale factors](how_to/fake_tau_scale_factors.md)
- [Derive Run 3 diboson jet-multiplicity scale factors](how_to/diboson_njets_scale_factors.md)

## Reference

Start with the [reference index](reference/README.md) when you need an exact
contract rather than a procedure. The corpus is divided by authority:

- [Supported wrappers and direct entrypoints](reference/entrypoints.md)
- [Analysis processor physics map](reference/analysis_processor.md)
- [Objects, selections, and triggers](reference/objects_selections_and_triggers.md)
- [Corrections, weights, and systematics](reference/corrections_weights_and_systematics.md)
- [Categories and observables](reference/categories_and_observables.md)
- [Sample roles and normalization](reference/sample_roles_and_normalization.md)
- [Data-driven estimation](reference/data_driven_estimation.md)
- [Shared topcoffea mechanism ownership](reference/shared_topcoffea_interfaces.md)
- [Production profiles, sample configuration, and option ownership](reference/production_configuration.md)
- [Histogram PKLs, sidecars, and transformed artifacts](reference/histogram_artifacts.md)
- [HistEFT software contract](reference/histeft.md)
- [Sumw2 policy and provenance](reference/sumw2.md)
- [Processing and fitting binning](reference/flexible_binning.md)
- [Plotting interfaces and configuration](reference/plotting.md)
- [Datacard and scaling interfaces and artifacts](reference/datacards_and_scalings.md)
- [Specialist analysis interfaces](reference/specialist_interfaces.md)
- [Run 2 b-tag scale-factor payloads](reference/btag_scale_factor_payloads.md)
- [Missing-parton payload schema](reference/missing_parton_payloads.md)

Source signatures, CLI parsers, registries, and configuration files remain the
machine-near authorities identified by these pages. The repository does not
currently publish generated API documentation.

## Explanation

- [Physics analysis flow](explanation/physics_analysis_flow.md)
- [Workflow architecture and actor boundaries](explanation/architecture.md)
- [Artifact and provenance model](explanation/artifacts_and_provenance.md)
- [Why sumw2 is a policy and artifact contract](explanation/sumw2.md)
- [Why processing and fitting binning are separate](explanation/flexible_binning.md)
- [Datacard, scaling, and EFTFit/Combine boundary](explanation/datacards_and_eftfit.md)
- [HistEFT data and coefficient model](explanation/histeft_data_model.md)
- [Missing-parton uncertainty model](explanation/missing_parton_uncertainties.md)

## Current specialist documentation

The specialist guides are maintained but are not steps in every analysis run:

- [Fake-tau scale-factor extraction](how_to/fake_tau_scale_factors.md)
- [Run 3 diboson jet-multiplicity scale factors](how_to/diboson_njets_scale_factors.md)
- [Missing-parton uncertainty model](explanation/missing_parton_uncertainties.md)
  and [payload maintenance](how_to/missing_parton_payloads.md)
- [Work Queue execution](how_to/work_queue.md)

## Historical material

These pages preserve context or reproduction information. They may require old
software, data, site services, or repository revisions and must not be read as
current TOP-26-006 operating instructions.

- [TOP-22-006 histogram and datacard reproduction](how_to/historical/top_22_006.md)
- [TOP-22-006 MC validation studies](how_to/historical/top_22_006_mc_validation.md)
- [Earlier Coffea training material](tutorials/historical/coffea_training_materials.md)
- [Legacy b-tag MC-efficiency study](how_to/historical/btag_mc_efficiency.md)
- [Legacy charge-flip measurement](how_to/historical/charge_flip_measurement.md)
- [Extreme-events study](how_to/historical/extreme_events_study.md)
- [Extreme-events visualization](how_to/historical/extreme_events_visualization.md)
- [Historical JES regrouping note](how_to/historical/jet_energy_uncertainty_regrouping.md)

## Repository-local pointers

Some source or payload directories may retain a short README that points back
to this documentation map. Those pointers are not separate authorities: task
procedures belong in `how_to/`, stable contracts in `reference/`, and design
rationale in `explanation/`.
