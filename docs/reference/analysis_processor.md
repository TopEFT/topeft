# Analysis processor physics map

`analysis/topeft_run2/analysis_processor.py` is the maintained main event
processor. The production chain is
`run_cr.sh` → `fullR3_run.sh` → `run_analysis.py` → `AnalysisProcessor`.
Wrappers and the direct CLI select or forward analysis policy; the processor
consumes that policy and produces selected, weighted histogram artifacts.

This reference maps the processor by physics responsibility. It is not
line-by-line pseudocode. Each row names a durable implementation authority and
the linked pages own detailed contracts.

## Authority boundaries

- `topeft` owns sample interpretation, concrete object and event policy,
  correction activation, category construction, observable selection, and
  output consumption.
- `topcoffea` owns reusable correction, selection, EFT-algebra, and histogram
  mechanisms. The [shared-interface index](shared_topcoffea_interfaces.md)
  records each boundary.
- Parameter files, channel registries, sample metadata, and packaged payloads
  own their configured values. The processor call site shows which authority
  is selected.
- Source establishes implemented behavior. It does not establish the physics
  motivation for a threshold, working point, region, or sample-role choice
  unless a separate authority is named.

## Physics execution map

| Processor section | Inputs and physics decision | Applicability and variations | Downstream effect and authority |
| --- | --- | --- | --- |
| Construction and allocation | Analysis mode, category groups, requested histograms, WC names, and output policy are bound into one processor contract. | All selected inputs; the constructor declares the possible object and weight variation namespaces. | Determines which outputs can be filled. `AnalysisProcessor.__init__` and the `run_analysis.py` constructor call site are authority. |
| Sample, era, and EFT interpretation | Dataset metadata resolves era, data/MC status, normalization, sample roles, and EFT treatment. | Per sample. These decisions determine which corrections, weights, and EFT branches can exist. | Controls normalization, overlap masks, coefficient preparation, and sumw2-at-SM behavior. The `AnalysisProcessor.process` dataset-parameter and EFT-preparation block and the sample schema are authority. |
| Collection setup and corrections | Raw event collections are prepared and calibrated before selection; corrected jets propagate to MET. | Era, data/MC status, collection, and payload family determine dispatch. Central and configured shifted collections are prepared. | Feeds every later object, category, observable, and dependent weight. The `AnalysisProcessor.process` collection-initialization and correction-dispatch block, correction modules, and shared factories are authority. |
| Leptons, taus, jets, and b tags | Loose/tight lepton roles, tau branches, corrected jet collections, forward objects, and tag collections are built. | Flavor, era, channel, data/MC status, and enabled analysis mode determine which branches apply. Shifted objects are rebuilt where required. | Changes multiplicities, event masks, weights, categories, and observables. The `AnalysisProcessor.process` object-variation loop and lepton/tau/jet/b-tag selection blocks plus selection registries are authority. |
| Trigger, cleaning, and overlap policy | Era/channel triggers, event filters, data primary-dataset exclusion, and sample-family role masks restrict the event population. | Trigger and cleaning are era/channel dependent; dataset exclusion is data-only; source-role masks are sample-family dependent. | Prevents double counting and defines accepted events before category filling. The `AnalysisProcessor.process` trigger, event-filter, primary-dataset, sample-role, and overlap-mask blocks plus event/sample-role registries are authority. |
| Event weights and scale factors | Normalization, pileup, lepton, trigger, b-tag, prefiring, generator, parton-shower, and scale components are assembled when applicable. | MC only except explicitly data-driven quantities; each component has its own era, sample, and object requirements. | Sets nominal yields and named weight variations. The `AnalysisProcessor.process` base-weight and kinematic-variation weight-attachment blocks plus correction interfaces are authority. |
| Object and event variations | Weight modifiers reuse nominal objects; object shifts rerun affected corrections, selections, categories, and observables. | Non-nominal processing occurs only when systematics are enabled and the source applies. | Produces comparable systematic templates. The `AnalysisProcessor.process` systematic-list setup and `for syst_var in syst_var_list` variation loop are authority. |
| CR, SR, and category construction | Multiplicity, charge, flavor, Z-window, jet, b-tag, tau, and forward-object masks are combined into registered categories. | Category registry, analysis mode, enabled features, and object variation determine membership. | Selects the channel-axis label and event partition filled downstream. The `AnalysisProcessor.process` category-mask construction blocks and `topeft/channels/ch_lst.json` are authority. |
| Data-driven source behavior | Fake and charge-flip weights and source roles are attached to the source histograms consumed by the transformation stage. | Applies only to the certified data/simulation sources and regions in the data-driven contract. | Enables later nonprompt and charge-flip products while preserving required variations. Processor call sites and data-driven policy/product modules are authority. |
| Observable construction and fill | Lepton, jet, MET, mass, angular, and channel-specific values are computed and filled with process, category, systematic, and EFT coordinates. | Observable availability depends on channel and enabled analysis; object shifts may recompute values. | Produces the maintained histogram corpus. The `AnalysisProcessor.process` observable-value construction and histogram-fill loops, `axes.py`, the category registry, and `HistEFT` are the split authorities. |
| Nominal, sumw2, and provenance output | Histogram families, statistical second-moment companions, and resolved metadata are returned. | Sumw2 storage follows the resolved policy and is distinct from physics nuisance variations. | Supplies plotting, card, transformation, and validation inputs. The `AnalysisProcessor.process` histogram-fill completion and `return hout` output boundary plus the sumw2 and artifact contracts are authority. |

## EFT source treatment

The processor validates whether a sample is treated as EFT-capable, ordinary
SM, or ignored for EFT purposes before coefficient preparation. That treatment
is sample policy owned by `topeft`. Polynomial coefficient ordering,
evaluation, and `HistEFT` storage are shared mechanisms owned by `topcoffea`.
The distinction prevents an ordinary sample from acquiring EFT semantics only
because the output container supports them.

See [HistEFT](histeft.md) and
[sample roles and normalization](sample_roles_and_normalization.md).

## Selection and overlap ordering

Corrected objects are defined before event categories. Trigger and cleaning
masks constrain the event set, while object-overlap cleaning and dataset/sample
overlap removal address different duplication risks. Category masks then
combine the accepted objects and events. The exact concrete definitions live
in [objects, selections, and triggers](objects_selections_and_triggers.md).

## Systematic propagation

Object shifts can change kinematics, object membership, event masks, category
membership, observables, and dependent weights. Weight-only variations retain
the nominal object and category view. Both paths reach the same histogram-fill
contract, which is why the systematic label can be consumed coherently by
plotting and datacards. Registered-but-unfilled legacy generator/flavor axes do
not constitute maintained output and are not part of this map.

See [corrections, weights, and systematics](corrections_weights_and_systematics.md).

## Output boundaries

The processor produces histogram content; it does not own the full production
campaign, data-driven materialization, plotting, card construction, or scaling
finalization. Statistical second moments are stored according to
[sumw2 policy](sumw2.md), while schemas, sidecars, and lineage belong to
[histogram artifacts](histogram_artifacts.md).

## Modification boundary

Changes must be made at the owning layer: sample roles and metadata, correction
dispatch and payload selectors, object/event policy, category and axis
registries, or shared `topcoffea` mechanisms. A direct change inside the fill
loop is not a substitute for extending those contracts. The contextual
procedures linked from this page are added in the practical documentation
layer.

## Representative event-flow anchor

For a maintained `2los_CRZ`/`invmass` request, the direct CLI resolves the
sample metadata and category group, the processor corrects the collections,
applies the era/channel trigger and cleaning masks, constructs the opposite-
sign CRZ category, computes the dilepton invariant mass, and fills the nominal
histogram. With `--do-systs`, applicable object shifts repeat the dependent
selection/category/observable steps, while weight modifiers reuse the nominal
object/category view. The example links concrete owners without asserting why
the region was chosen:

- category: [`ch_lst.json`](../../topeft/channels/ch_lst.json)
- observable: [`axes.py`](../../topeft/modules/axes.py)
- entrypoint and defaults:
  [`run_analysis.py`](../../analysis/topeft_run2/run_analysis.py)
- correction/selection consumers:
  [`analysis_processor.py`](../../analysis/topeft_run2/analysis_processor.py)

Use [categories and observables](../how_to/categories_and_observables.md) to
extend the category or observable, and
[corrections, weights, and systematics](../how_to/corrections_weights_and_systematics.md)
to extend the variation flow.
