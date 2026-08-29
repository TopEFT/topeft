# Missing-parton uncertainties

The missing-parton uncertainty accounts for a modeling limitation in selected
top-EFT signal events: a reconstructed event category may not contain every
parton needed for the intended generator-level interpretation. The correction
is represented as a shape-and-rate nuisance, derived outside ordinary analysis
production and consumed when card templates are built.

This page explains the physics and software boundaries. Use the
[payload-maintenance how-to](../how_to/missing_parton_payloads.md) for commands
and the [payload reference](../reference/missing_parton_payloads.md) for the
installed files, public schema, selection rules, and source/test authorities.

## Why the uncertainty is separate

The nominal histogram answers: “what weighted event yield did the configured
analysis select?” The missing-parton model asks a different question: “how
does that yield change when signal events without the full partonic content are
treated as a modeling uncertainty?” Embedding the answer directly in the
processor would mix a derived modeling policy with event selection and would
make its provenance difficult to inspect.

Instead, the workflow has three owners:

- the derivation layer measures the effect from frozen source artifacts;
- the installed payload records the era-, process-, category-, and bin-aware
  result;
- `DatacardMaker` evaluates that payload on the exact fitting view used for
  each card template.

This separation lets the source measurement, public schema, and statistical
consumer evolve under explicit compatibility rules.

## Data and authority flow

The source workflow produces one correction file per supported era family:
Run 2 and Run 3 are distinct authorities. The derivation reads source
histograms and supporting ROOT information, validates the relevant channels
and axes, calculates the directional changes, and writes a candidate payload.
Installation is a separate action because it changes packaged runtime data.

At card production:

1. `make_cards.py` selects a coherent histogram family and exact fitting view.
2. `DatacardMaker` resolves the appropriate missing-parton payload for the
   artifact era.
3. The consumer evaluates the payload for the selected signal process,
   category, and physical bin.
4. The resulting up/down factors are written into the corresponding card
   templates and nuisance definition.

The payload does not select a campaign, channel list, or fitting binning. The
card consumer supplies those resolved identities. Conversely, the card code
does not own the numerical correction; it validates and evaluates the packaged
payload.

## Physics definition

The derivation separates events that contain the expected generator-level
parton content from the inclusive signal selection. For each supported
process, category, and bin it constructs directional variations describing the
effect of the missing component.

Two ingredients must remain distinct:

- **shape information** records how the fractional missing component changes
  across bins;
- **rate information** records the normalization component associated with the
  selected process/category scope.

The stored correction combines these ingredients into consumer factors. It is
not an arbitrary bin-by-bin smoothing and it is not derived from the nominal
template alone.

### Residual amount and stored fraction

The derivation starts from inclusive and full-parton yields on compatible
axes. Their residual identifies the amount attributed to the missing-parton
component. The stored fractional representation must remain well-defined when
the inclusive yield is small, empty, negative, or otherwise unsuitable for a
naive division.

Invalid or non-finite inputs are a derivation failure, not a reason to publish
an apparently neutral factor. The validator reconstructs the expected
directional behavior from the candidate payload and rejects a payload whose
fractions, factors, bin association, or finite-value contract is inconsistent.

### Consumer factors

The public payload provides the quantities required to construct up/down
template factors. The consumer applies them to the selected nominal signal
template on the resolved physical binning. A factor therefore has meaning only
with its era, process, category, and bin association.

The exact branch and field names are part of the
[public payload schema](../reference/missing_parton_payloads.md#current-schema).
They must not be reinterpreted from filename patterns or derivation logs.

## Terminal-bin semantics

The terminal physical bin needs explicit treatment because it can include an
open-ended tail. ROOT overflow content, the last visible bin, and the selected
fit interval are not interchangeable concepts.

The derivation and validator establish the physical edges and flow convention
before comparing or reconstructing the correction. The card consumer later
evaluates the correction on the same resolved fitting mapping. A payload that
has the expected number of values but a different terminal-bin meaning is not
compatible.

This is also why the missing-parton model follows flexible-binning rules:
processing axes, fitting edges, and flow aggregation must be physically
compatible before a payload can be applied. See
[flexible binning](flexible_binning.md).

## Forward-category correction

Forward categories have a dedicated correction boundary because their event
composition and final observable differ from ordinary categories. The
derivation preserves the category identity and validates the forward mapping
rather than applying a process-global factor.

The durable mapping belongs to current channel and axis authorities, not to a
campaign wrapper. If category definitions or the final distribution change,
the payload must be rederived and consumer validation updated; renaming a
payload does not preserve compatibility.

## Process, era, and correlation policy

Missing-parton records are process-aware. A correction derived for one signal
process cannot be substituted for another solely because their category names
match. Run 2 and Run 3 payloads are also separate, selected from artifact-era
evidence rather than an output tag.

The nuisance correlation policy is part of the card-model contract. It must be
consistent with how the numerical records are grouped by era and process. The
payload schema supplies the numerical information; it does not silently choose
a broader or narrower statistical correlation than the card configuration.

## Source, payload, and consumer boundaries

### Derivation owner

`analysis/topeft_run2/missing_parton.py` and its supporting tools own source
selection, calculation, candidate output, and derivation diagnostics. A dry
run can validate the intended source set without writing. A write must target
a fresh scratch payload unless a separate overwrite action is authorized.

### Payload owner

`topeft/data/missing_parton/` owns the installed Run 2 and Run 3 correction
files and their identity/provenance record. Installation is not implied by a
successful derivation. Both era files are treated as one maintained payload
family because partial replacement could make runtime selection inconsistent.

### Consumer owner

`topeft.modules.datacard_tools.MissingParton` and `DatacardMaker` own payload
selection, evaluation on selected fitting bins, template construction, and
fail-closed behavior. They do not own the source derivation or installation.

## Provenance and reproducibility

A reproducible candidate records the exact source inputs and verifies them
before numerical interpretation. Source paths or campaign names are not enough:
the derivation freezes identities and hashes, and validation reopens the
candidate to check schema and semantic content.

The public payload must be non-empty, finite, correctly associated, and
reconstruct the derived directional behavior. Structural ROOT readability
alone is insufficient. The maintenance how-to preserves the order:
freeze sources, validate without writing, create a fresh candidate, validate it
independently, and only then consider a separate installation action.

## Interaction with sumw2

The source histograms used for the derivation are weighted. Required sumw2
companions provide the statistical content needed for validation and must agree
with the artifact's resolved policy and content manifest.

The payload does not carry or replace histogram sumw2. It carries a derived
modeling correction. A source artifact missing a required companion is
statistically incomplete for the intended derivation and fails at the source
validation boundary. See [the sumw2 model](sumw2.md).

## Historical context

Earlier payloads were replaced after bounded source and downstream studies
identified terminal-bin and category-specific limitations. Those comparisons
remain historical evidence for why the current contract is stricter; they are
not a recipe for mixing old and current artifacts or for claiming numerical
identity across incompatible bins.

The historical quantitative results are retained in repository history and
the accepted diagnostic evidence for the payload refresh. Current maintenance
uses the public schema, installed identity, and consumer tests documented in
the reference. Reopening the historical production study is not required for a
normal payload selection or card run.

## Extension and failure boundary

A change to the physics definition, supported process/category set, public
schema, era-selection rule, or correlation meaning crosses more than one owner.
It requires a separately scoped implementation decision with derivation,
payload, consumer, and test updates. Documentation alone must not bridge those
owners with a compatibility assumption.

The current system fails closed on missing payloads, unsupported eras,
malformed schema, non-finite or wrongly associated content, incompatible bin
semantics, or a consumer request outside the certified scope. The
[payload-maintenance how-to](../how_to/missing_parton_payloads.md) routes each
failure to its owning layer.
