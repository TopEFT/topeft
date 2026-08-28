# Software reference

This reference describes the maintained analysis interfaces, configuration
schemas, and artifact contracts. Use it when you need an exact default, field,
accepted value, failure condition, or source owner. The source paths named on
each page remain authoritative for callable signatures and executable behavior.

For a guided analysis, start from the [documentation index](../README.md). For
instructions that change or run something, use the linked how-to guide rather
than treating an option table as a recipe.

## Reference map

| Topic | What it specifies |
| --- | --- |
| [Entrypoints and wrappers](entrypoints.md) | Supported production, transformation, plotting, card, and scaling commands and their responsibility boundaries |
| [Analysis processor physics map](analysis_processor.md) | Source-derived physics execution sections, inputs, applicability, systematic propagation, ownership, and outputs |
| [Objects, selections, and triggers](objects_selections_and_triggers.md) | Maintained object roles, working-point ownership, triggers, cleaning, overlap, and event-selection consequences |
| [Corrections, weights, and systematics](corrections_weights_and_systematics.md) | Concrete correction dispatch, payload/working-point policy, central and varied behavior, and downstream effects |
| [Categories and observables](categories_and_observables.md) | CR/SR and category implementation, observable authority, and processing/fitting views |
| [Sample roles and normalization](sample_roles_and_normalization.md) | Sample metadata, role sets, ttgamma policy, normalization, and specialist sum-of-weights boundary |
| [Data-driven estimation](data_driven_estimation.md) | Nonprompt and charge-flip source roles, transformations, variations, and product authority |
| [Shared topcoffea interfaces](shared_topcoffea_interfaces.md) | Cross-repository mechanism versus analysis-policy ownership index |
| [Production configuration](production_configuration.md) | Sample cfg/JSON inputs, production profiles, option overlays, current NDSkim reachability, and output defaults |
| [Histogram artifacts and provenance](histogram_artifacts.md) | PKL/sidecar pairs, schemas, content manifests, lineage, transformation, merge, and validation contracts |
| [HistEFT and SparseHist](histeft.md) | Current histogram API, consumers, and pickle compatibility, with future replacement-parity design clearly separated |
| [Sumw2 policy](sumw2.md) | Modes, default, selectors, resolved policy, companion naming, provenance, and failure conditions |
| [Flexible binning](flexible_binning.md) | Processing and fitting schemas, family defaults, exact channel-name overrides, and exact aggregation |
| [Plotting](plotting.md) | Wrapper/direct-CLI defaults, configuration owners, binning views, and output controls |
| [Datacards and scalings](datacards_and_scalings.md) | Card CLI, `DatacardMaker`, card/template pairs, selected WCs, scaling records, and final channel mapping |
| [Specialist interfaces](specialist_interfaces.md) | Artifact inspection and developer-facing interfaces that support focused maintenance work |
| [B-tag scale-factor payloads](btag_scale_factor_payloads.md) | Packaged UL files, current consumers, and provenance limits |
| [Missing-parton payloads](missing_parton_payloads.md) | Installed Run 2/Run 3 payload schema, selection, overrides, compatibility, and test authority |

## Find a contract by interface type

### Wrappers and CLIs

- Production: `run_cr.sh`, `fullR3_run.sh`, and `run_analysis.py` in
  [entrypoints](entrypoints.md), with fields in
  [production configuration](production_configuration.md).
- Transformations: `run_data_driven.py` in
  [entrypoints](entrypoints.md), with product schemas in
  [histogram artifacts](histogram_artifacts.md).
- Plotting: `run_plotter.sh` and `make_cr_and_sr_plots.py` in
  [plotting](plotting.md).
- Cards and scaling finalization: `make_cards.py` and
  `datacards_post_processing.py` in
  [datacards and scalings](datacards_and_scalings.md).
- Inspection: `inspect_histeft_pkl.py` in
  [specialist interfaces](specialist_interfaces.md).

### Modules, classes, and symbols

- `analysis_processor.AnalysisProcessor` physics execution sections:
  [analysis processor physics map](analysis_processor.md); constructor and
  configuration interface: [production configuration](production_configuration.md).
- Object/event policy and shared selection boundaries:
  [objects, selections, and triggers](objects_selections_and_triggers.md).
- Correction dispatch, event weights, and variation propagation:
  [corrections, weights, and systematics](corrections_weights_and_systematics.md).
- Category masks, observables, and views:
  [categories and observables](categories_and_observables.md).
- Sample metadata/roles and normalization:
  [sample roles and normalization](sample_roles_and_normalization.md).
- Nonprompt and charge-flip products:
  [data-driven estimation](data_driven_estimation.md).
- `production_sample_profile`: [production configuration](production_configuration.md).
- `sumw2_policy`: [sumw2 policy](sumw2.md).
- `axes` and `axis_binning`: [flexible binning](flexible_binning.md).
- `nominal_schema`, `histogram_artifact`, and `data_driven_products`:
  [histogram artifacts](histogram_artifacts.md).
- `HistEFT` and `SparseHist`: [HistEFT API](histeft.md).
- Plotter context, coverage, diagnostics, and rendering surfaces:
  [plotting](plotting.md).
- `datacard_tools`, including `DatacardMaker`:
  [datacards and scalings](datacards_and_scalings.md).

### Schemas, artifacts, configuration, and constants

- PKL keys, sidecars, manifests, lineage, and transformed-product schemas:
  [histogram artifacts](histogram_artifacts.md).
- Sumw2 modes, selectors, provenance versions, and resolved-policy fields:
  [sumw2 policy](sumw2.md).
- Processing/fitting registry fields and exact channel-name overrides:
  [flexible binning](flexible_binning.md).
- Sample cfg/JSON identity and production sample certification:
  [production configuration](production_configuration.md).
- Plot metadata: [plotting](plotting.md).
- `ch_lst.json`, cards, templates, selected WCs, and scaling JSON:
  [datacards and scalings](datacards_and_scalings.md).
- Installed correction inputs: [B-tag payloads](btag_scale_factor_payloads.md)
  and [missing-parton payloads](missing_parton_payloads.md).
- Cross-repository shared mechanisms and analysis policy:
  [shared topcoffea interfaces](shared_topcoffea_interfaces.md).

Statuses used on these pages are `public supported` for analyst-facing
interfaces, `developer-facing` for maintained extension contracts, `internal
extension` for implementation seams useful only within their owner, and
`archival operator record` for executable historical evidence that is not a
supported current interface.

## Authority convention

The curated pages explain stable semantics so a reader does not have to infer
them from implementation details. They do not mechanically duplicate every
Python or shell signature. For exact callable parameters, read the named
function, class, parser, or wrapper help block in the checked-out source. When a
curated statement and current executable source differ, stop and resolve the
contradiction before production.

## API publication

The repository does not currently publish generated API pages, and this
documentation does not introduce a generator. That tooling limitation is
independent of the curated coverage above. Source docstrings and annotations
remain machine-near authority where they are substantive; these pages supply
the stable contract when machine-near documentation is shallow.
