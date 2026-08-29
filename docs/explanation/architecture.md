# Analysis workflow architecture

This page is the authority for responsibilities and boundaries in the current
TOP-26-006 `topeft` workflow. It explains why the actors are separate, what
each owns, and what evidence must cross each boundary. It is not an option
reference or a list of commands.

For a guided run, use the [analysis tutorial](../tutorials/analysis_workflow.md).
For a task, use the [how-to index](../how_to/README.md). For exact CLI, symbol,
schema, and failure contracts, use the [reference index](../reference/README.md).

## The authority chain

The workflow deliberately separates five kinds of authority:

1. **Input authority** identifies samples and files through sample JSONs and
   sample cfg bundles.
2. **Orchestration authority** chooses a production plan, output namespace,
   environment identity, and resume boundary.
3. **Analysis authority** selects events, applies corrections and weights, and
   fills histogram families.
4. **Artifact authority** publishes histogram content with enough provenance
   for a later consumer to reject an incompatible input.
5. **Statistical-input authority** selects exact fit views, writes individual
   cards and templates, and maps physical channels into the namespace used at
   the EFTFit/Combine boundary.

No filename or wrapper is allowed to collapse those authorities into one
implicit convention. A wrapper may automate an owner; it does not replace that
owner's source contract.

## Inputs and production orchestration

### Sample JSON and sample cfg layers

**Why they exist.** A sample JSON describes one dataset or sample: files,
metadata, year, cross section, process identity, and related loader input. A
sample cfg, also called a sample bundle, selects a set of sample JSONs for one
analysis request.

**Consumes and produces.** The cfg resolves to sample JSONs; the sample loader
resolves those into the active sample universe consumed by `run_analysis.py`.
The production-sample-profile helpers derive and certify the resulting cfg,
dataset, and process identity.

**Owns.** These files own input selection and sample metadata. The runtime
branches in `fullR3_run.sh` identify which maintained NDSkim bundles are
reachable for a given year and region.

**Does not own.** A cfg does not own event selection, histogram definitions,
sumw2 policy, a campaign output namespace, or a fit topology. A cfg's presence
in `input_samples/cfgs/` is not evidence that current production selects it.

**Boundary and extension.** A direct user may provide one explicit cfg or
sample JSON authority. A maintained wrapper change must keep cfg selection in
the existing owner and update the sample-profile tests. See
[production configuration](../reference/production_configuration.md).

### Production profile

**Why it exists.** A production profile is a named plan in `run_cr.sh`. It
groups blocks that must share campaign, environment, state, and output
invariants. `run3_full` is the maintained complete Run 3 plan; `rebin_fine` is
a separate specialist plan.

**Consumes and produces.** It consumes its declared profile name, a fresh or
exactly resumable output namespace, an environment request, and block options.
It produces a frozen plan/state record and delegates each block to
`fullR3_run.sh`.

**Owns.** It owns orchestration order, block identity, state transitions,
namespace collision checks, resume evidence, and the separate source/nonprompt
process lifecycle.

**Does not own.** It does not define selections, corrections, histogram axes,
sample JSON contents, or statistical models. A profile must not duplicate
those registries.

**Boundary and extension.** A profile extension changes the block plan and its
state invariants together. It must preserve unique output identities and update
the profile/resume tests. See [production operations](../how_to/production.md)
and the [`run_cr.sh` reference](../reference/entrypoints.md#analysistopeft_run2run_crsh).

### `run_cr.sh`

**Why it exists.** `run_cr.sh` is the supported high-level campaign wrapper.
It turns a production profile into a repeatable, fail-closed sequence rather
than a set of manually coordinated shell commands.

**Consumes and produces.** It consumes a profile, output directory, campaign
tag, optional current environment archive, and resume/dry-run intent. It
produces the campaign state and the source and transformed artifacts required
by each block.

**Owns.** It owns plan freezing, environment and repository identity checks,
fresh-output protection, ambiguous-interruption boundaries, block sequencing,
and deferred nonprompt invocation after the processor child exits.

**Does not own.** It does not own cfg internals or the `run_analysis.py`
argument semantics that it forwards through `fullR3_run.sh`.

**Current checkout boundary.** The maintained script changes into the
correction-lib-managed `topeft/analysis/topeft_run2` checkout and reads Git
identity from that managed repository. It therefore orchestrates that checkout,
not an arbitrary clone or the caller's current working directory. Making the
wrapper portable is an executable-interface change outside this documentation
contract.

**Failure boundary.** A missing or mismatched state, reused fresh namespace,
invalid environment archive, unexpected partial output, or incompatible resume
request fails before the wrapper declares a block complete.

### `fullR3_run.sh`

**Why it exists.** This is the supported middle layer for one bounded analysis
block. It converts year, region, input, histogram, and output choices into one
`run_analysis.py` command.

**Consumes and produces.** It consumes Run 2/Run 3 year and CR/SR choices,
optional cfg/sample overrides, histogram variables, nonprompt intent, and
forwarded lower-level options. It produces a resolved command; when executed,
that command produces the block's histogram artifact.

**Owns.** It owns maintained NDSkim cfg selection, year/region expansion,
wrapper-level defaults, conflict checks, and transparent forwarding.

**Does not own.** It does not own a multi-block campaign, resume state, event
processing, or artifact schemas. A direct call therefore moves campaign
bookkeeping to the user.

**Boundary and extension.** A new wrapper option must be owned here or
forwarded, never both. New cfg logic stays in the centralized selection branch.

### `run_analysis.py`

**Why it exists.** `run_analysis.py` is the independently supported direct CLI
for one analysis request and the common lower-level entrypoint used by the
wrappers.

**Consumes and produces.** It consumes one cfg/sample expression, CLI and
optional YAML options, executor/resource settings, years, categories,
histogram families, and output identity. It resolves the active sample and
sumw2 policies, runs the processor, and publishes a source PKL plus sidecar.
Depending on nonprompt options it may also run or defer the transformed
artifact step.

**Owns.** It owns CLI/YAML precedence, preflight resolution, executor
construction, processor invocation, output naming, and artifact publication.

**Does not own.** It does not choose a maintained campaign matrix, protect a
multi-block namespace, or resume a failed campaign stage. A direct caller must
record the exact inputs, options, years, categories, histogram list, executor,
and output identity.

**Boundary and extension.** New supported options belong in its parser and
resolved configuration path, with validation before processor execution. See
the [entrypoint reference](../reference/entrypoints.md#analysistopeft_run2run_analysispy).

## Analysis and histogram production

### `AnalysisProcessor` and the processor layer

**Why it exists.** `AnalysisProcessor` turns NanoEvents into selected and
weighted analysis categories and fills the nominal, systematic, EFT, and
sumw2 histogram families.

**Consumes and produces.** It consumes resolved sample metadata, analysis-mode
flags, categories, corrections, weights, EFT coefficients, histogram-family
definitions, and a resolved sumw2 policy. It produces an in-memory histogram
mapping returned to `run_analysis.py`.

**Owns.** It owns event selection, category masks, correction and weight
application, observable computation, EFT treatment, and physical fills on the
processing axes.

**Does not own.** It does not own campaign state, output directories, sidecar
publication, fit-bin aggregation, plotting style, card topology, or EFTFit
workspace construction.

**Failure boundary and extension.** Analysis-mode and category configuration
must resolve before event work. New fills must use the authoritative axis and
nominal-family contracts and must propagate any required sumw2 sibling. The
[processor reference](../reference/production_configuration.md#analysis_processoranalysisprocessor)
lists its stable developer surfaces.

### Processing binning

**Why it exists.** Processing binning is the physical dense axis stored when
events are filled. It retains enough resolution for later plotting and exact
fit aggregation.

**Consumes and produces.** `axes.info` supplies the family definition;
`axis_binning.make_processing_axis` creates the fill axis. The produced PKL
contains that physical axis for nominal, EFT, systematic, and sumw2 content.

**Owns and does not own.** The registry owns numeric definitions. The processor
uses them but does not redefine them. Processing binning does not select a fit
view. Changing it changes newly produced artifacts and requires new PKLs.

### Sumw2 companions and policy

**Why they exist.** A nominal weighted bin stores the sum of event weights; a
sumw2 companion stores the sum of squared event weights needed for statistical
uncertainties. Selective storage limits artifact size without silently omitting
content required by consumers.

**Consumes and produces.** The resolver consumes a mode, active sample
universe, histogram families, and optional selector rules. It produces a
concrete target policy. The processor creates `<family>_sumw2` siblings for
those targets; the sidecar records both policy provenance and the actual
content manifest.

**Owns.** `sumw2_policy.py` owns modes, the current `production` default,
resolution, schema-v2 serialization, schema-v1 readback, and identity checks.

**Does not own.** A mode name is not a provenance schema, and the resolved
policy is not proof that the companion content was written. Consumers validate
the manifest against their own requirements.

See [why sumw2 is a contract](sumw2.md), the
[sumw2 how-to](../how_to/sumw2.md), and the
[sumw2 reference](../reference/sumw2.md).

## Artifact and transformation boundary

### Histogram PKL

**Why it exists.** The compressed PKL transports the processor's histogram
mapping, including physical axes, nominal and systematic content, EFT
coefficients, and any sumw2 companions.

**Consumes and produces.** `run_analysis.py` serializes the processor result;
plotting, transformation, merge, and card consumers load a compatible family.

**Owns and does not own.** The PKL owns the histogram payload. It does not, by
itself, prove sample identity, transformation lineage, or compatibility with
another file. Those claims require its sidecar and content readback.

### Metadata sidecar and provenance

**Why it exists.** The adjacent `.metadata.json` file makes the artifact's
identity and compatibility evidence inspectable before a large payload is
composed or consumed.

**Consumes and produces.** Artifact helpers derive it from the resolved source
contract and actual output. It records source/family identity, axes, Wilson
coefficient order, production sample profile, sumw2 policy/content, and any
transformation lineage.

**Owns.** The artifact module owns sidecar schemas, validation, merge
compatibility, lineage, and atomic PKL/sidecar publication.

**Does not own.** A caller cannot invent provenance fields to make an
incompatible payload pass. A sidecar also cannot certify a consumer-specific
requirement without content validation.

See [artifacts and provenance](artifacts_and_provenance.md) and the
[artifact reference](../reference/histogram_artifacts.md).

### `run_data_driven.py` and transformed nonprompt artifacts

**Why it exists.** Nonprompt estimation combines prompt-subtracted source
processes into a data-driven product after the heavy processor step. The
maintained campaign runs it in a fresh process so processor memory and
transformation recovery have separate boundaries.

**Consumes and produces.** It consumes a compatible source PKL and sidecar,
the resolved data-driven product contract, and required sumw2 companions. It
produces `_np.pkl.gz` plus a transformed sidecar with lineage.

**Owns.** It owns the nonprompt/flips transformation, streaming versus legacy
materialization choice, output validation, and transformed artifact
publication.

**Does not own.** It does not own source-event processing, prompt sample
authority, plotting, card creation, or the decision to equate source and
transformed products. They are deliberately distinct.

**Failure boundary.** Missing prompt authority, incomplete execution coverage,
missing required companions, incompatible sidecars, or output collision fails
before a transformed product is certified. See the
[nonprompt how-to](../how_to/nonprompt.md).

## Plotting and validation boundary

### `run_plotter.sh`

**Why it exists.** The plotting wrapper supplies common filename-driven
region convenience, Run 2/Run 3 year aliases, blinding defaults, and option
forwarding.

**Consumes and produces.** It consumes an artifact path, output directory,
years, region intent, variables, and forwarded plotter options. It constructs
one `make_cr_and_sr_plots.py` command and produces no independent metadata
registry.

**Owns and does not own.** It owns shell convenience and validation of its own
arguments. It delegates artifact compatibility, channel resolution, binning,
plot configuration, and rendering to the direct plotter.

### `make_cr_and_sr_plots.py`

**Why it exists.** This is the direct supported plotting and analysis-
validation entrypoint.

**Consumes and produces.** It consumes one coherent artifact family, years,
channel/variable choices, plotting metadata, and a processing or fitting view.
It produces plots, yield summaries, and negative-contribution diagnostics.

**Owns.** It owns artifact loading, channel authority resolution, mixed-energy
rejection, late bin view, region context, visual output, and diagnostics.

**Does not own.** It does not produce or modify card templates, artifact
provenance, physical processing axes, or production campaign state.

### `axes.py` and `axis_binning.py`

`axes.py` owns processing and fitting definitions. `axis_binning.py` owns their
schema validation, exact physical channel-name lookup, common-edge resolution,
flow-aware aggregation map, and histogram rebinning. The fitting `channels`
mapping uses exact keys; only `make_cards.py --ch-lst` uses analyst-supplied
regular expressions for channel selection.

Plotting defaults to the processing view. Cards default to the fitting view.
Both use the same resolver so a copied bin table cannot drift. See
[why the views are separate](flexible_binning.md).

## Card, scaling, and fit boundary

### `make_cards.py`

**Why it exists.** This is the direct supported card-production CLI; there is
no maintained general campaign card wrapper.

**Consumes and produces.** It consumes positional coherent PKLs or one list
file, channel and variable selections, Wilson-coefficient choices, binning and
coverage policy, and card options. It produces individual text/ROOT card-
template pairs and `scalings-preselect.json`. In the normal selection path it
also writes `selectedWCs.txt`. With `--use-selected`, it reads the supplied JSON
but does not copy that file into the output directory, so the operator must
place the reviewed selection there before finalization.

**Owns.** It owns CLI input resolution, merge validation, selection, card
configuration, and invocation of `DatacardMaker`.

**Does not own.** It does not own campaign matrix provenance, final `chN`
ordering, card combination, or workspace construction.

### `datacard_tools` and `DatacardMaker`

**Why they exist.** `datacard_tools` contains the reusable producer logic for
validated artifact loading, systematic configuration, exact late binning,
templates, text cards, selected Wilson coefficients, and preselection scaling
records.

**Consumes and produces.** `DatacardMaker` consumes the merged histogram
mapping and selected configuration. It produces the per-physical-channel
artifacts and scaling rows requested by `make_cards.py`.

**Owns.** It owns the template/card content, systematic pairs, negative-
nominal handling, statistical-uncertainty policy, fitted-axis mapping, and the
producer-owned `{channel, process, parameters, scaling}` record.

**Does not own.** It does not own deterministic final topology selection or
the external `chN` workspace namespace.

### Individual card/template pair

Each physical channel has a text card and corresponding ROOT template. Their
physical channel and distribution identify the selected fit view. A pair is
not interchangeable with `selectedWCs.txt` or either scaling JSON; the complete
set defines the statistical handoff.

### `selectedWCs.txt`

This file records the Wilson coefficients selected for the produced card set.
It is copied into the finalized directory and consumed together with the cards
and scaling payload. It does not contain the EFT scaling coefficients itself.

### `scalings-preselect.json`

This producer output contains physical-channel scaling records before final
topology selection. Multiple records for a physical channel/process may be
valid. Their process, parameters, and coefficient payload remain producer-
owned through finalization.

### `datacards_post_processing.py`

**Why it exists.** The finalizer selects one declared topology and establishes
the deterministic namespace shared with the later EFTFit/Combine card order.

**Consumes and produces.** It consumes a datacard directory already containing
individual cards/templates, `selectedWCs.txt`, and `scalings-preselect.json`,
plus exactly one topology selector. With `-a` it uses the full current topology
from `ch_lst.json`. It creates the selected output subdirectory and writes
`scalings.json`.

**Owns.** It owns physical category-to-distribution predicates, sorted physical
channel ordering, `ch1`, `ch2`, … assignment, selected-file copying, scaling-
row filtering, and channel relabeling.

**Does not own.** It neither reads nor creates `combinedcard.txt`; it does not
combine cards or build a workspace.

### Physical channel to `chN` mapping and `scalings.json`

The finalizer sorts selected physical channel names and maps the item at index
`i` to `ch{i+1}`. Every matching scaling record keeps all producer-owned fields
except `channel`, which is relabeled. A missing final record means no external
EFT morph for that exact channel/process.

### EFTFit and Combine

**Why they exist.** EFTFit and Combine own the statistical-model boundary after
`topeft` has produced and finalized its individual inputs.

**Consumes and produces.** They consume the selected individual cards and
templates, `selectedWCs.txt`, and compatible ordered `scalings.json`; they later
combine cards, create `combinedcard.txt`, and construct the workspace.

**Owns and does not own.** They own card combination and workspace/likelihood
construction. `topeft` does not define a current copy-paste external command,
and EFTFit does not retroactively own production, histogram, or fitting-bin
registries in this repository.

See [datacards and the external boundary](datacards_and_eftfit.md), the
[card how-to](../how_to/datacards_and_scalings.md), and the
[card/scaling reference](../reference/datacards_and_scalings.md).

## Repository ownership

`topeft` owns analysis processors, selections, workflow entrypoints,
histogram/artifact policy, plotting, cards, and scaling finalization.
`topcoffea` owns shared libraries, corrections, common weights and utilities.
An installed `topcoffea` payload or API may be a runtime dependency without
making `topcoffea` the owner of `topeft` cards, axes, or workflows.

This documentation describes the current `topeft` side of that boundary. It
does not change or restate `topcoffea` APIs and it does not introduce a second
registry for cross-repository state.

## Current and historical automation

Current TOP-26-006 workflows use the supported entrypoints described above.
Historical TOP-22-006 procedures live only under
[`docs/how_to/historical`](../how_to/historical/README.md).

`run_make_cards_run3_yawen_matrix.sh` is a DATACARD023 archival operator
record: it binds site/user paths, an exact campaign matrix, hashes, branch and
environment assumptions, and an input that predates the final `ptll` schema.
It is not a maintained public wrapper or a second region/binning authority.
This documentation may classify it, but moving, deleting, generalizing, or
requalifying the runnable script is a separate source-control decision.
