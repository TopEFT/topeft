# Histogram artifacts and HistEFT tutorial

This tutorial explains what a `topeft` histogram artifact contains, how EFT
information differs from an ordinary histogram yield, and how to inspect an
artifact without treating its filename as proof of provenance. It complements
the broader [analysis workflow tutorial](analysis_workflow.md): start there if
you do not yet know where histogram production sits in TOP-26-006.

The practical goal is to answer four questions about a PKL and its sidecar:

1. Which samples, channels, observables, and systematic variations does it
   contain?
2. Which content is scalar and which content retains EFT coefficient
   dependence?
3. Which statistical `sumw2` companions were selected and actually published?
4. Is the artifact a processor source or a transformed data-driven product,
   and what may safely consume or combine it?

Commands use an activated environment with the intended `topeft` and
`topcoffea` revisions. A dry run or `--pretend` request does not produce a PKL.

## 1. From weighted events to an EFT-aware histogram

An ordinary weighted histogram stores a sum of event weights in each physical
bin. A typical analysis bin is identified by categorical coordinates such as
process, channel, and systematic variation plus one dense numerical coordinate
such as jet multiplicity or transverse momentum.

For example, a conceptual bin might mean:

```text
process = ttH_private2023
channel = one particular 2lss signal-region category
systematic = nominal
njets = 4
```

The processor decides whether an event belongs in that bin and computes its
complete event weight. The histogram object owns accumulation; it does not own
the selection or correction definitions that produced the values.

### Why HistEFT stores more than one number

For an EFT signal, the expected yield depends on Wilson coefficients. With one
coefficient `c`, a bin can be written schematically as

```text
yield(c) = a0 + a1*c + a2*c^2
```

With several Wilson coefficients, the bin also contains mixed quadratic terms.
`HistEFT` accumulates the coefficient array needed to evaluate this polynomial
later. The same processed events can therefore be evaluated at the Standard
Model point or another supported coefficient point without rerunning event
selection.

`HistEFT` owns coefficient accumulation, evaluation, histogram algebra, and
serialization compatibility. It does not own sample loading, event selection,
artifact sidecars, plotting, fitting-bin choice, or final card topology. See
[the HistEFT data model](../explanation/histeft_data_model.md) for this boundary
and the [HistEFT reference](../reference/histeft.md) for exact methods and
pickle-compatibility requirements.

### HistEFT and SparseHist in the same artifact

The workflow can carry both EFT-aware and scalar content:

- `HistEFT` represents one-dimensional EFT-polynomial content.
- scalar `SparseHist` content represents fixed yields, including data and
  non-EFT processes.
- two-dimensional histogram families remain scalar `SparseHist` objects.
- a `sumw2` companion is a separate scalar second-moment histogram at
  `WC = 0`; it is not another EFT polynomial.

This separation is deliberate. A data yield does not acquire Wilson-coefficient
dependence merely because it shares an observable with an EFT signal.

## 2. Understand the histogram coordinates

When you inspect a histogram, distinguish categorical axes from the dense
physical axis.

- **`process`** identifies the resolved sample/process contribution after the
  producer's grouping rules.
- **`channel`** identifies an analysis category. A channel name is a physics
  label, not the later `chN` card-combination label.
- **`systematic`** distinguishes nominal content and supported variations.
- **`appl`** identifies application or signal/control content used by
  data-driven transformations when that axis is present.
- **the dense axis** is the observable itself, such as `njets`, `lj0pt`,
  `ptz`, or `ptll`.

Sparse categorical axes let one object contain many processes and categories.
The dense axis supplies the ordered numerical bins shown in a plot or used by
a card. Processing bin edges are physically stored in the artifact. A fitting
view may later aggregate them exactly, but it cannot invent a non-nested set of
edges. See [flexible binning](../explanation/flexible_binning.md).

Systematic content is another coordinate, not a separate claim about
provenance. Before comparing nominal and an `Up` or `Down` label, list the
labels actually present and consult the producer/consumer contract for whether
a complete pair is required.

## 3. Inspect a bounded production request

For learning, use `fullR3_run.sh --dry-run` to inspect one resolved command.
Do not edit the maintained `run_cr.sh` profile arrays to make a tutorial small.

From `analysis/topeft_run2`:

```bash
./fullR3_run.sh \
  -y 2023 \
  -t histogram_tutorial \
  --sr \
  --hist-vars njets \
  --category-groups 2l \
  -p /absolute/path/to/tutorial_output \
  -x futures --nworkers 1 --nchunks 1 \
  --pretend --np-postprocess=skip \
  --dry-run
```

This command demonstrates wrapper resolution only:

- `-y 2023` selects one Run 3 year;
- `--sr` selects signal-region configuration;
- `--hist-vars njets` asks for one small histogram family;
- `--category-groups 2l` limits category construction;
- the futures/executor options describe a bounded direct request;
- `--pretend` would stop after input discovery if the shell dry-run were
  removed;
- `--np-postprocess=skip` avoids requesting a transformed artifact in this
  source-artifact exercise;
- `--dry-run` prints the delegated `run_analysis.py` command and does not run
  Python.

The output should show the resolved years, region, histogram list, input cfg
bundle, output path/name, and delegated direct command. That printed command is
the point at which wrapper-owned decisions become explicit.

For a single-sample exercise, `fullR3_run.sh --sample-json FILE` replaces the
maintained cfg bundle for that invocation. A current 2023 EFT signal example is

```text
../../input_samples/sample_jsons/signal_samples/ND_SRskim2023/ttH_NDSkim_2023.json
```

Use an explicit sample override only after inspecting its year, process-axis
name, file access, and Wilson-coefficient metadata. The sample JSON is input
authority; it does not become a production profile.

See the [production how-to](../how_to/production.md) for wrapper modification
and the [entrypoint reference](../reference/entrypoints.md) for exact CLI
boundaries.

## 4. Recognize the published artifact pair

An authorized real request without `--dry-run` or `--pretend` publishes a
histogram payload and adjacent sidecar:

```text
<output_dir>/<outname>.pkl.gz
<output_dir>/<outname>.pkl.gz.metadata.json
```

The PKL holds histogram objects. The JSON sidecar holds the identity and
compatibility record needed before a consumer loads, transforms, or combines
them. Treat the two paths as one artifact.

The sidecar records, as applicable:

- artifact kind and schema/layout versions;
- source and histogram-family identity;
- physical axes and Wilson-coefficient order;
- production sample/profile contract;
- requested and resolved sumw2 policy;
- a manifest of the companion content actually present;
- data-driven applicability and, for transformed artifacts, lineage.

The separation lets a consumer inspect compatibility information without
first materializing a potentially large PKL. The consumer still validates the
record against serialized content. A matching filename, output tag, shape, or
number of bins is insufficient evidence.

Exact schema values and publication/readback functions are in the
[artifact reference](../reference/histogram_artifacts.md). The reason for a
separate sidecar is explained in
[artifacts and provenance](../explanation/artifacts_and_provenance.md).

## 5. Use the read-only inspector

Start at the repository root and inspect the helper interface:

```bash
python analysis/topeft_run2/inspect_histeft_pkl.py --help
```

List an artifact's top-level keys and bounded axis labels:

```bash
python analysis/topeft_run2/inspect_histeft_pkl.py \
  /path/to/output.pkl.gz \
  --max-labels 10
```

Focus on one family and request a simple nominal summary when the helper can
derive it unambiguously:

```bash
python analysis/topeft_run2/inspect_histeft_pkl.py \
  /path/to/output.pkl.gz \
  --hist njets \
  --max-labels 10 \
  --yield-summary
```

The inspector reports the top-level object and keys, histogram-like object
types, axes, bounded categorical labels, discoverable Wilson coefficients, and
optional simple nominal yield/variance summaries. It does not certify that the
artifact is suitable for a particular plot or card; those consumers impose
additional requirements.

Use the maintained helper before writing ad hoc pickle-loading snippets. It
avoids printing complete arrays and provides one consistent first view. When
you need programmatic evaluation, slicing, grouping, arithmetic, or pickle
parity details, use the [HistEFT reference](../reference/histeft.md) rather than
copying an API catalog into this tutorial.

## 6. Read the nominal split-family layout

Current one-dimensional source artifacts use nominal container schema 2 and
the `split_sibling_v1` layout. For one family, the possible top-level keys are:

```text
<family>__scalar_nominal   scalar nominal content, when present
<family>__eft_nominal      EFT-polynomial nominal content, when present
<family>_sumw2             selected scalar second moment, when present
```

The scalar and EFT nominal siblings cover non-overlapping process content. A
family need not have all three keys:

- no scalar processes means no scalar nominal sibling;
- no EFT processes means no EFT nominal sibling;
- a companion appears only when the resolved sumw2 policy selects concrete
  content for that family.

The old unsplit one-dimensional `<family>` producer key is not the current
schema-2 output. Consumers may construct a local materialized view after
validating the siblings; that view is not a new producer layout.

Two-dimensional families do not use the scalar/EFT sibling split. They remain
scalar `SparseHist` objects under `<family>` and may have an optional
`<family>_sumw2` companion.

This layout keeps fixed scalar yields, EFT polynomial coefficients, and
statistical second moments distinct while allowing consumer-local composition.
The exact nominal-schema and compatibility rules live in the
[artifact reference](../reference/histogram_artifacts.md) and
[HistEFT reference](../reference/histeft.md).

## 7. Interpret nominal yields and EFT evaluations

For a scalar process, a nominal bin is a fixed weighted yield. For an EFT
process, the stored coefficients define a function of Wilson coefficients.
At the Standard Model point, all non-SM coefficients are zero. A nonzero point
changes only processes carrying compatible EFT coefficient content.

Before comparing evaluations:

1. Select the same process, channel, systematic label, and physical bins.
2. List the Wilson coefficients actually stored; do not assume a coefficient
   exists because it appears in another campaign.
3. Evaluate the Standard Model point as a baseline.
4. Change one supported coefficient point and compare like with like.
5. Keep statistical second moments separate from the EFT polynomial value.

An unknown Wilson coefficient should fail rather than be silently ignored.
Coefficient order must remain compatible across fragments and transformations.
The exact `eval`, `as_hist`, slicing, grouping, addition, and pickle behavior is
curated in [the software reference](../reference/histeft.md).

## 8. Understand the sumw2 companion

For event contributions `w_i`, nominal content and statistical second moment
answer different questions:

```text
nominal bin content = sum_i(w_i)
statistical second moment = sum_i(w_i^2)
```

Signed Monte Carlo weights can cancel in the nominal sum while their squared
contributions remain positive. Reconstructing `sumw2` from a nominal yield or
from a Poisson count would therefore change the statistical meaning.

The maintained companion contract is scalar Standard-Model (`WC = 0`) second
moment content for concrete dataset/process/family targets selected by policy.
Nonzero-WC quartic variance is outside this contract.

Keep these facts separate during inspection:

- the **mode** is the named policy request;
- the **default** defines what omission means;
- the **resolved policy** records concrete selected targets;
- the **provenance schema** defines how that resolution is serialized;
- the **content manifest** records what companion content is physically
  present.

A mode name in metadata does not prove that required companion content exists.
Consumers recompute requirements and compare them to the payload. See the
[sumw2 explanation](../explanation/sumw2.md),
[selection/extension how-to](../how_to/sumw2.md), and
[policy reference](../reference/sumw2.md).

## 9. Follow a source through data-driven transformation

`run_data_driven.py` consumes a validated processor source and creates a
distinct transformed nonprompt artifact. Its sidecar records lineage and a
transformation contract; `_np.pkl.gz` is not merely a renamed copy.

Data-driven applicability is family-specific. The source-wide policy may
enable nonprompt or flips, while one histogram family lacks the application
regions needed for a particular product. The transformed sidecar distinguishes:

- which application-region labels existed in the source;
- which products were applicable to the family;
- which nonprompt or flips processes were generated;
- which sumw2 processes the transformed family requires.

An unknown `isAR_*` label is not automatically accepted. When a product is
applicable and required, missing generated content fails validation. When it is
not applicable, the transformer does not invent it.

For programmatic inspection, read the sidecar with
`read_histogram_sidecar(...)` and validate the pair with
`validate_histogram_artifact(...)`. Then compare the manifest's required
companion processes with the actual `<family>_sumw2` process axis. Exact
transformation fields, functions, and compatibility behavior belong to the
[artifact reference](../reference/histogram_artifacts.md) and
[HistEFT contract](../reference/histeft.md); operational recovery belongs to
the [nonprompt how-to](../how_to/nonprompt.md).

## 10. Understand what plotting does with the artifact

`make_cr_and_sr_plots.py` consumes one or more validated, coherent artifacts.
It resolves process grouping and channel authority, evaluates HistEFT content
at the requested coefficient point, applies the requested processing or
fitting view, and builds figures and diagnostic reports.

Plotting does not mutate the input PKL. It also does not make provenance true:
artifact identity, family compatibility, Wilson-coefficient order, and sumw2
requirements are checked before composition. Mixed Run 2 and Run 3 selections
and ambiguous producer channel presets fail.

Start with a wrapper dry run:

```bash
./analysis/topeft_run2/run_plotter.sh \
  -f /path/to/output_np.pkl.gz \
  -o /absolute/path/to/plots \
  -y run3 --sr \
  --variables njets \
  --dry-run
```

Use the [plotting how-to](../how_to/plotting.md) for real operations and
extension points, and the [plotting reference](../reference/plotting.md) for
CLI defaults and metadata ownership.

## 11. Debug from the contract outward

### No PKL was written

- Confirm that neither the shell wrapper nor Python request was a dry run.
- `--pretend` resolves inputs and stops before event processing.
- Read the final printed output path and output name rather than guessing from
  a tag.
- Check the executor error before assuming artifact publication failed.

### A histogram or category is missing

- Confirm the family was included in `--hist-vars`/`--hist-list`.
- Confirm the requested category group exists in the active `ch_lst.json`
  region block.
- Check whether `--skip-sr` or `--skip-cr` removed the target region.
- List actual top-level keys and channel labels with the inspector.

### A process is missing

- Confirm that its sample JSON is reachable through the selected cfg or direct
  input expression.
- Inspect the JSON's process-axis identity rather than inferring it from the
  filename.
- Distinguish a missing producer process from a plotting-group configuration.

### A systematic is missing

- Confirm systematic production was requested.
- List actual labels before selecting an `Up` or `Down` variation.
- Use consumer errors to distinguish an optional absent process from a
  required incomplete shape pair.

### EFT evaluation fails

- List the stored Wilson-coefficient names.
- Confirm the selected process has an EFT nominal sibling.
- Check coefficient ordering/identity when combining fragments.
- Treat an unknown coefficient failure as useful evidence, not a reason to
  edit the pickle.

### Sumw2 validation fails

- Compare requested/resolved policy, provenance schema, content manifest, and
  actual companion processes separately.
- Do not infer variance from nominal content.
- Add or select a policy mode through the producer and regenerate the source;
  do not inject a companion into an existing PKL.

### Two artifacts will not merge

- Compare source identity, axes, Wilson-coefficient order, production sample
  contract, nominal layout, sumw2 policy/content, and transformation lineage.
- Resolve overlapping contributions instead of bypassing collision checks.
- Do not use matching filenames or shapes as compatibility evidence.

## 12. Deeper contracts and next steps

This tutorial intentionally does not duplicate the exhaustive software API.
Use the document that owns the next question:

| Question | Owner |
| --- | --- |
| How do I run the complete workflow? | [Analysis workflow tutorial](analysis_workflow.md) |
| How do I run or modify one production route? | [Production how-to](../how_to/production.md) |
| What are the exact PKL/sidecar schemas and artifact functions? | [Histogram-artifact reference](../reference/histogram_artifacts.md) |
| What are the exact HistEFT/SparseHist methods and pickle requirements? | [HistEFT reference](../reference/histeft.md) |
| Why are payload and sidecar separate? | [Artifacts and provenance](../explanation/artifacts_and_provenance.md) |
| Why does HistEFT store polynomial coefficients? | [HistEFT data model](../explanation/histeft_data_model.md) |
| How do I select or extend sumw2 storage? | [Sumw2 how-to](../how_to/sumw2.md) |
| How do I inspect/change fitting aggregation? | [Flexible-binning how-to](../how_to/flexible_binning.md) |
| How do I produce plots? | [Plotting how-to](../how_to/plotting.md) |

## Glossary

- **artifact:** a PKL plus its adjacent metadata sidecar.
- **categorical axis:** labels such as process, channel, systematic, or
  application region.
- **dense axis:** ordered numerical observable bins.
- **EFT:** effective field theory, parameterized here by Wilson coefficients.
- **HistEFT:** histogram object that stores a quadratic EFT polynomial in each
  physical bin.
- **nominal:** the central, non-varied systematic label.
- **PKL:** gzip-compressed Python-serialized histogram mapping (`.pkl.gz`).
- **processing binning:** physical dense axis stored by the producer.
- **fitting binning:** exact downstream aggregation view used by cards and
  optionally plotting.
- **provenance:** structured evidence describing source identity, policy,
  compatibility, and lineage.
- **sidecar:** adjacent JSON record for an artifact.
- **SparseHist:** scalar histogram representation used for fixed and
  two-dimensional content.
- **sumw2:** sum of squared event contributions used for statistical second
  moments.
- **transformed artifact:** nonprompt/flips output with recorded lineage to a
  validated source.
- **Wilson coefficient:** EFT parameter whose value changes the evaluated
  polynomial yield.
