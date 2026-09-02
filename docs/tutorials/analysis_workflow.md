# Analysis workflow tutorial

This tutorial follows the maintained TOP-26-006 path from sample selection to
the artifacts handed to EFTFit and Combine, the downstream statistical tools.
It is written for a HEP analyst who knows why events are selected and
histogrammed but does not yet know which `topeft` component owns each step.

Commands are shown from `analysis/topeft_run2` in an activated environment
containing the intended `topeft` and `topcoffea` revisions. Paths beginning
with `/absolute/path/to/` are placeholders. Inspect commands and choose fresh
output locations before starting a real campaign.

## Terms used below

- A **control region (CR)** is selected to constrain or validate backgrounds;
  a **signal region (SR)** is selected to test the target signal hypothesis.
  The regions can share software while retaining different event categories.
- **Coffea** is the columnar Python analysis framework used here.
  **NanoEvents** is Coffea's structured event view of NanoAOD records; the
  processor reads physics objects and event fields from that view.
- **Nonprompt backgrounds** contain selected leptons that do not come directly
  from the prompt hard interaction, such as leptons from heavy-flavor decays or
  misidentified objects. A **charge-flip background** arises when a lepton's
  electric charge is reconstructed incorrectly. Data-driven transformations
  estimate these contributions where the configured application regions allow.
- **Effective field theory (EFT)** extends the Standard Model with additional
  interactions. **Wilson coefficients (WCs)** are the parameters multiplying
  those interactions; HistEFT stores enough information to evaluate histogram
  yields at supported WC values.
- An analysis **category** is an event class such as `2lss` (two same-sign
  light leptons), `3l`, `4l`, or `2los_CRZ` (a two-lepton opposite-sign Z
  control-region group). Example observables include `njets` (jet
  multiplicity), `lj0pt` (the largest transverse momentum among two-object
  four-vector sums drawn from selected leptons and jets, also including cleaned
  taus when tau-h processing is enabled), `ptz` (Z-candidate transverse
  momentum), and `ptll` (dilepton transverse momentum for explicitly mapped
  final off-Z categories). `o0pt`, not `lj0pt`, is the leading single-object
  transverse momentum. Their exact owners and channel mappings are in the
  [flexible-binning reference](../reference/flexible_binning.md).

## 1. The workflow in one physical picture

A collider analysis begins with events from data or simulation. The processor
selects events, computes weights, and accumulates those weighted events into
histogram bins. The resulting histograms are then checked with plots and
converted into statistical-model inputs. `topeft` keeps these steps separate
so that the code making a physics selection is not also responsible for
campaign bookkeeping or for constructing a Combine workspace.

```text
sample JSONs / sample cfg
    -> production profile and wrappers
    -> run_analysis.py and AnalysisProcessor
    -> source histogram PKL + metadata sidecar
    -> run_data_driven.py
    -> transformed nonprompt PKL + metadata sidecar
    -> plotting and validation
    -> make_cards.py
    -> individual cards/templates + selectedWCs.txt + scalings-preselect.json
    -> datacards_post_processing.py
    -> selected cards/templates + selectedWCs.txt + scalings.json
    -> EFTFit / Combine
```

The arrows describe responsibility and data flow, not one monolithic command.
Each stage validates the artifact contract it consumes.

## 2. Meet the actors and artifacts

Read this section once before using the commands. The short definitions tell
you what each actor owns and, equally importantly, what it does not own.

### Inputs and orchestration

- **Sample JSON and sample cfg.** A sample JSON describes one dataset: its
  files and analysis metadata such as process identity, year, and normalization
  inputs. A sample cfg is a bundle or expression that points to several sample
  JSONs so they can be loaded together. These files answer *which events are
  available*; they do not define selections, histogram axes, or campaign
  resume state. `fullR3_run.sh` selects maintained cfg bundles, while a direct
  caller may provide a JSON or cfg explicitly. See
  [production configuration](../reference/production_configuration.md).

- **Production profile.** A profile is a named `run_cr.sh` orchestration plan,
  such as `run3_full`. It exists to freeze a reproducible matrix of blocks,
  environment identity, output names, and resume state. It consumes wrapper
  options and delegates work; it does not redefine channels, corrections,
  histogram families, or sumw2 policy. See the
  [architecture explanation](../explanation/architecture.md).

- **`run_cr.sh`.** This is the maintained campaign-level wrapper. It expands a
  production profile, checks a fresh output namespace or an exact resume, and
  invokes `fullR3_run.sh` for each block. It produces campaign state plus
  validated source and transformed artifacts. It does not process events
  itself. See the [production how-to](../how_to/production.md#run-a-maintained-run_crsh-campaign)
  and [entrypoint reference](../reference/entrypoints.md).

- **`fullR3_run.sh`.** This lower-level wrapper resolves Run 2 or Run 3 years,
  CR/SR mode, the maintained sample cfg, histogram-family arguments, and one
  `run_analysis.py` command. It produces that command and, when actually run,
  the delegated command's artifacts. It does not create campaign state or
  coordinate several blocks. See
  [run one block](../how_to/production.md#run-one-block-with-fullr3_runsh).

- **`run_analysis.py`.** This is the supported direct analysis CLI. It consumes
  a sample JSON/cfg expression and CLI or YAML options, validates the requested
  sample universe and policies, constructs `AnalysisProcessor`, runs the
  selected executor, and publishes the source artifact. It does not choose a
  campaign matrix, protect a multi-block namespace, or resume a campaign. See
  the [direct-run how-to](../how_to/production.md#run-run_analysispy-directly).

### Processing and histogram artifacts

- **`AnalysisProcessor` and the processor layer.** The processor owns event
  object selection, analysis categories, event weights, systematic variations,
  and histogram filling. It consumes NanoEvents plus resolved configuration
  and returns histogram accumulators. It does not own executor scheduling,
  atomic artifact publication, plotting, or card topology. The broader actor
  boundary is described in [workflow architecture](../explanation/architecture.md).

- **Source histogram PKL.** A PKL is the gzip-compressed Python serialization
  of the processor's histogram dictionary. It contains nominal, systematic,
  EFT-polynomial, and policy-selected statistical content. It exists so
  downstream tools can work without rereading every event. A source PKL is not
  self-authenticating and is not the final nonprompt product. Follow the
  [histogram-artifact tutorial](histogram_artifacts.md) for guided inspection.

- **Metadata sidecar and provenance.** The adjacent
  `<name>.pkl.gz.metadata.json` records the evidence needed to interpret and
  combine a PKL: artifact kind, source/family identity, axes, Wilson-coefficient
  order, production sample contract, sumw2 policy/content, and transformation
  lineage. Artifact helpers own publication and validation. A filename or
  matching bin count does not replace this record. See
  [artifacts and provenance](../explanation/artifacts_and_provenance.md) and the
  [artifact reference](../reference/histogram_artifacts.md).

- **Sumw2 companion.** If events contribute weights `w` to a bin, the nominal
  content contains `sum(w)` while the statistical second moment contains
  `sum(w^2)`. Negative weights make it especially unsafe to infer the second
  quantity from the first. A selected `<family>_sumw2` companion stores the
  Standard-Model (`WC = 0`) second moment for a concrete policy target. The
  sumw2 registry selects which companions exist; it does not change nominal
  physics content and does not provide nonzero-WC quartic variances. See
  [why sumw2 is a policy](../explanation/sumw2.md), the
  [sumw2 how-to](../how_to/sumw2.md), and the
  [exact policy reference](../reference/sumw2.md).

- **Processing and fitting binning.** Processing edges are the physical dense
  axis stored in the source PKL. Fitting edges are an exact downstream
  aggregation of those bins, usually coarser to make a statistically useful
  fit. A family may have a fitting default and exact channel-name overrides;
  those keys are not regular expressions. Plotting defaults to processing
  binning and cards default to fitting binning. Neither view permits a
  non-nested interpolation. See the
  [binning explanation](../explanation/flexible_binning.md),
  [modification how-to](../how_to/flexible_binning.md), and
  [reference](../reference/flexible_binning.md).

- **`run_data_driven.py` and the transformed artifact.** This entrypoint
  consumes a validated source PKL/sidecar and applies the configured nonprompt
  and charge-flip transformation where each histogram family is applicable. It
  produces a distinct `_np.pkl.gz` artifact with a transformed sidecar and
  lineage back to its source. It does not overwrite the source or decide which
  plots/cards should be made. See the
  [nonprompt how-to](../how_to/nonprompt.md).

### Validation and statistical-model inputs

- **Plotting and validation layer.** `run_plotter.sh` is reusable shell
  automation over the direct `make_cr_and_sr_plots.py` CLI. They consume a
  coherent artifact family, apply the requested processing or fitting view,
  group processes, and render distributions and diagnostic reports. They do
  not modify the PKL or produce cards. See the
  [plotting how-to](../how_to/plotting.md) and
  [plotting reference](../reference/plotting.md).

- **`make_cards.py`.** This is the supported direct card interface. It consumes
  one coherent final histogram family, selects variables/channels/systematics,
  applies exact fitting aggregation, and delegates card construction to
  `DatacardMaker`. It produces individual card/template pairs,
  `selectedWCs.txt`, and `scalings-preselect.json`. It does not combine cards,
  finalize the selected topology, or build a workspace. See the
  [card how-to](../how_to/datacards_and_scalings.md) and
  [card reference](../reference/datacards_and_scalings.md).

- **Individual card/template pair.** Each selected physical channel and
  distribution has a text datacard and matching ROOT template. The card records
  processes, rates, and nuisance definitions; the ROOT file holds the nominal
  and varied shapes. The pair is one statistical-model input, not a combined
  analysis card.

- **`selectedWCs.txt`.** This records the Wilson coefficients selected during
  card production. It gives downstream tooling the parameter ordering/context
  associated with the templates and scaling payload. It does not contain the
  bin-by-bin EFT polynomial itself.

- **`scalings-preselect.json`.** `make_cards.py` writes this producer-owned
  collection of physical-channel/process scaling records before the final
  topology is selected. It may contain records for channels that are not in
  the final fit. Its payload must be filtered and relabeled, not replaced by a
  process-global number.

- **`datacards_post_processing.py`.** This topology/scaling finalizer consumes
  a datacard directory that already contains individual pairs,
  `selectedWCs.txt`, and `scalings-preselect.json`. With one topology selector,
  it copies the selected pairs and WCs, derives a deterministic channel order,
  relabels matching scaling records, and writes `scalings.json`. It does not
  read or create `combinedcard.txt`. See
  [finalize scalings](../how_to/datacards_and_scalings.md#finalize-the-current-full-topology).

- **Physical channel to `chN` mapping.** Before finalization, a channel has a
  physics-facing name that encodes its category and final distribution. The
  finalizer sorts the selected physical names and assigns `ch1`, `ch2`, and so
  on. This makes the scaling namespace deterministic and aligned with later
  card combination. It is a mapping, not a loss of the physical meaning
  carried by the selected card/template names.

- **`scalings.json`.** This is the selected, ordered EFT-scaling payload. Each
  retained record preserves its producer-owned process, coefficient, and bin
  content while replacing the physical channel label with the corresponding
  `chN`. Absence of a record means no external EFT morph for that exact
  channel/process; it is not an implicit normalization rule.

- **EFTFit and Combine.** These downstream tools own individual-card
  combination, `combinedcard.txt`, workspace construction, and statistical
  inference. They consume the selected cards/templates, `selectedWCs.txt`, and
  compatible `scalings.json`. Correction-lib does not own those downstream
  operations. See [the handoff explanation](../explanation/datacards_and_eftfit.md).

## 3. Choose one of the three production routes

The three routes reach the same direct analysis CLI but move responsibility
between automation and the caller.

| Route | Choose it when | Automation provided | You must record or manage |
| --- | --- | --- | --- |
| `run_cr.sh` | Running or resuming a maintained multi-block campaign | Profile block matrix, fresh namespace, environment archive, state/resume, source/nonprompt stage checks | Selected profile, output root, campaign tag, reviewed frozen plan, repository revisions |
| `fullR3_run.sh` | Running one supported Run 2/Run 3 block or inspecting wrapper resolution | Maintained cfg selection, years/region expansion, output-name construction, one direct command | Campaign grouping, state/resume, exact printed command, output identity, downstream lifecycle |
| `run_analysis.py` | Running one focused request or developing the direct CLI | Input loading, validation, executor/processor construction, source publication | Exact input/options, years/categories/histograms, executor, output namespace, commit/environment, every downstream step |

### Route A: maintained campaign with `run_cr.sh`

The current wrapper is bound to the correction-lib managed checkout: it changes
to that checkout's analysis directory and records the Git commit from that same
repository. It does not operate on an arbitrary clone or derive repository
identity from the caller's working directory. Use this route only in the
managed workspace for which it is configured; use a lower-level route when
working from another checkout.

Inspect the frozen plan first:

```bash
./run_cr.sh \
  --production-profile run3_full \
  --output-dir /absolute/path/to/fresh_run3_campaign \
  --campaign-tag run3_campaign \
  --dry-run
```

Remove `--dry-run` only after reviewing the resolved blocks and confirming that
the fresh output directory does not exist. A later resume must use the exact
frozen state; do not edit the state file. The detailed procedure and extension
invariants are in the [production how-to](../how_to/production.md).

### Route B: one block with `fullR3_run.sh`

This representative command resolves one 2022 SR block and prints the delegated
analysis command:

```bash
./fullR3_run.sh \
  -y 2022 -t tutorial_2022 --sr \
  --hist-vars njets lj0pt ptz ptll ptz_wtau lt \
  --do-np --defer-np \
  -p /absolute/path/to/output \
  --dry-run
```

Remove `--dry-run` only after inspecting the input bundle, output name, and
forwarded options. This route has no campaign state: record the printed direct
command and manage source/nonprompt completion yourself. See
[how to modify `fullR3_run.sh`](../how_to/production.md#run-one-block-with-fullr3_runsh).

### Route C: direct `run_analysis.py`

The direct route can resolve a focused request without processing events:

```bash
python run_analysis.py \
  ../../input_samples/cfgs/NDSkim_2022_background_samples.cfg \
  --executor futures --years 2022 --nworkers 8 \
  --hist-list njets lj0pt ptz ptll lt \
  --category-groups 2los_CRZ \
  --outpath /absolute/path/to/output --outname tutorial_2022 \
  --pretend
```

Remove `--pretend` only when the resolved files, categories, histogram list,
executor, and output identity are the request you intend to run. The direct
route does not reconstruct the wrapper decisions you omitted. Record the exact
input expression, optional `--options FILE`, resolved option values, repository
commit, environment identity, and downstream plan. Recognized values loaded
from that YAML file are applied after argument parsing and replace the
corresponding parser-derived values; do not assume a duplicate CLI value has
final precedence. The
[entrypoint reference](../reference/entrypoints.md) owns exact defaults and
failure boundaries.

The remainder begins at the common source publication and is route-neutral.
An ordinary clone can continue with the reviewed Route B command without
`--dry-run`, or with Route C after removing `--pretend` and recording the
omitted orchestration decisions. In the configured correction-lib workspace,
Route A can instead automate the multi-block source/nonprompt lifecycle. All
three routes ultimately reach the same `run_analysis.py` and processor
boundary.

## 4. Reach a source artifact from any route

Choose the launch command that matches your checkout. In an ordinary clone,
execute the reviewed `fullR3_run.sh` command without `--dry-run`, or the direct
`run_analysis.py` command without `--pretend`. In the managed workspace,
`run_cr.sh` delegates each profile block through `fullR3_run.sh`. In every
case, `run_analysis.py` constructs `AnalysisProcessor` and publishes the same
kind of source artifact.

The source publication is a pair:

```text
<outname>.pkl.gz
<outname>.pkl.gz.metadata.json
```

Treat these as one artifact. Before combining fragments or advancing a stage,
the consumer checks source identity, histogram families, physical axes,
Wilson-coefficient order, production sample contract, and sumw2 policy/content.
Follow the [histogram-artifact tutorial](histogram_artifacts.md) to inspect the
pair, and use the [artifact reference](../reference/histogram_artifacts.md) when
you need exact schema fields.

At this point, distinguish the two binning decisions. The PKL stores processing
edges. A later plotting or card consumer may request fitting edges only when
they are an exact aggregation. Channel-specific fitting selection uses exact
channel names, never regex matching. Also keep `ptz` and `ptll` distinct: only
the explicitly mapped final off-Z categories use `ptll`.

## 5. Produce the transformed nonprompt artifact

The managed Route A profile requests deferred nonprompt production and starts
`run_data_driven.py` in a fresh process after each source job. Route B, as
shown above, also requests deferred production, but the caller must run the
printed follow-up command and record completion. A direct Route C caller must
explicitly choose `--do-np --np-postprocess=defer` to use the same boundary,
then run the printed follow-up. Each route produces the same transformed pair:

```text
<outname>_np.pkl.gz
<outname>_np.pkl.gz.metadata.json
```

The transformed sidecar records lineage and family-specific applicability. A
family without an applicable data-driven region does not gain an invented
product; an applicable required product must be present. The `_np` suffix is a
convenient name, not the evidence for those facts. Direct and recovery recipes
are in the [nonprompt how-to](../how_to/nonprompt.md).

Sumw2 companions move through this boundary according to the resolved policy
and consumer requirements. A missing required companion is an error; neither
the transformer nor later tools substitutes a Poisson estimate from nominal
content.

## 6. Validate distributions before making cards

Use plotting to inspect yields, shapes, statistical uncertainty, systematic
variations, and whether the chosen aggregation makes physical sense. A minimal
wrapper invocation is:

```bash
./run_plotter.sh \
  -f /path/to/final_np.pkl.gz \
  -o /absolute/path/to/plots \
  -y run3 --sr \
  --variables lj0pt ptz ptll lt \
  --channel-output merged --dry-run
```

Review the printed direct plotter command, then remove `--dry-run` when ready.
The plotter consumes one coherent Run 2 or Run 3 artifact family; mixed
Run 2/Run 3 inputs fail. Plot metadata comes from
`topeft/params/cr_sr_plots_metadata.yml`, with bounded recovery from one
coherent producer channel preset. See the [plotting how-to](../how_to/plotting.md)
before changing wrapper forwarding or metadata, and the
[plotting reference](../reference/plotting.md) for exact defaults.

Plotting defaults to the stored processing view. Request fitting binning when
you want to inspect the exact view that cards will consume. Presentation-only
integer rebinning is a third operation and does not redefine either registry.

## 7. Create individual cards and scaling preselection

Use a coherent final `_np.pkl.gz` family for ordinary card production:

```bash
python make_cards.py /path/to/final_np.pkl.gz \
  --out-dir /absolute/path/to/cards \
  --var-lst lj0pt ptz ptll ptz_wtau lt \
  --ch-lst '^2lss_.*' '^3l_.*' '^4l_.*' \
  --binning fitting \
  --year-coverage-policy error
```

Here `--ch-lst` belongs to the card CLI's channel selection interface. It does
not change the separate `axes.info[family]["fitting"]["channels"]` contract,
whose keys are exact channel names.

`make_cards.py` validates/merges the input family, selects variables and
channels, applies exact fitting aggregation to nominal, sumw2, systematic, and
EFT content, and writes:

- individual `ttx_multileptons-*.txt` cards;
- matching `ttx_multileptons-*.root` templates;
- `selectedWCs.txt`;
- `scalings-preselect.json`.

Inspect this directory before finalization. The campaign-specific matrix
wrappers are archival operator records, not supported replacements for the
direct CLI. Use the
[datacard/scaling how-to](../how_to/datacards_and_scalings.md) for selection and
extension procedures and the
[reference](../reference/datacards_and_scalings.md) for exact artifact fields.

## 8. Select the final topology and write `scalings.json`

For the full current analysis topology:

```bash
python datacards_post_processing.py /absolute/path/to/cards -a
```

The directory must already contain the individual cards/templates,
`selectedWCs.txt`, and `scalings-preselect.json`. The `-a` selector reads the
full topology from `topeft/channels/ch_lst.json`. The finalizer derives each
physical card-channel name, sorts the selected names, maps them to `ch1`,
`ch2`, and so on, and writes a selected output directory.

Every matching scaling record keeps its producer-owned process, parameter,
coefficient, and bin payload; only its channel label becomes the deterministic
`chN`. The final file is `scalings.json`. The historically named
`ptz-lj0pt_withSys` directory does not imply that every selected channel uses
`ptz` or `lj0pt`.

## 9. Cross the EFTFit/Combine boundary

Carry the complete selected directory across the repository boundary:

- individual text cards and ROOT templates;
- `selectedWCs.txt`;
- final `scalings.json`.

EFTFit later combines the individual cards, creates `combinedcard.txt`, and
constructs the workspace used by Combine. Therefore `combinedcard.txt` is
neither an input nor an output of `datacards_post_processing.py`. A final
scaling record is interpreted for one `chN`/process/bin combination; an absent
record means no external EFT morph for that exact channel/process.

Read [the datacard and EFTFit boundary](../explanation/datacards_and_eftfit.md)
before diagnosing a channel-order or workspace mismatch. This repository does
not own a universal copy-paste EFTFit command.

## 10. Where to continue

One representative physics path ties the layers together: select the
`2los_CRZ` category group and `invmass` observable, let the processor apply
era/sample corrections and event masks, and retain nominal plus applicable
systematic content in the artifact. Plotting may display the processing view;
cards may select an exact fitting view. The category remains owned by
`ch_lst.json`, the observable by `axes.py`, and the view by `axis_binning.py`.
See the [processor physics map](../reference/analysis_processor.md) before
following the operating links below.

- To inspect HistEFT objects and source/transformed artifacts, continue with
  the [histogram-artifact tutorial](histogram_artifacts.md).
- To change one supported operation, choose the relevant page from the
  [how-to index](../how_to/README.md).
- To look up a CLI, module, schema, or artifact, use the
  [reference index](../reference/README.md).
- To understand why responsibilities are separated, use the
  [explanation map](../README.md#explanation).
- For the published predecessor rather than the current workflow, use the
  explicitly [historical TOP-22-006 guide](../how_to/historical/top_22_006.md).

## Current production example

For maintained source production, choose an explicit public profile such as
`run3_full`, a fresh absolute output directory, and a campaign tag. The public
matrix also includes `run2_full`, their `_CR` variants, and the two combined
Run2/Run3 variants. The no-argument form is only the legacy `run2_full` alias;
`rebin_fine` is a specialist legacy profile.

Recover from observed state rather than applying a generic retry. A known
component-local failure preserves its component evidence and may permit an
independent component to continue. A shared blocker stops remaining work.
After an ambiguous interruption, inspect durable campaign state, the output
namespace, and native Work Queue logs before selecting any follow-up. This
tutorial does not claim that a particular campaign has completed successfully.
