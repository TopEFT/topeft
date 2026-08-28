# Select or change processing and fitting binning

Processing binning is the physical dense axis stored in a histogram PKL.
Fitting binning is an exact late-aggregation view used by cards and, when
requested, plots. Keeping them separate allows fine production artifacts to
support validation and several coarser downstream views without interpolation.

The authoritative owners are:

- `topeft/modules/axes.py`: typed processing definitions and optional fitting
  defaults/exact channel-name overrides;
- `topeft/modules/axis_binning.py`: schema validation, exact resolution,
  aggregation maps, flow handling, physical-edge comparison, and histogram
  rebinning;
- `AnalysisProcessor`: builds/fills only processing axes;
- `make_cr_and_sr_plots.py`: defaults to processing, or resolves one common
  exact view for grouped physical channels;
- `DatacardMaker`: defaults to fitting and resolves each physical card channel.

Plot metadata and campaign/card wrappers may select a view. They are not
numeric bin-edge authorities. See the
[flexible-binning explanation](../explanation/flexible_binning.md) for the
design and the [reference](../reference/flexible_binning.md) for exact schemas
and functions.

## Select a downstream view

To inspect or plot the physical stored resolution:

```bash
python analysis/topeft_run2/make_cr_and_sr_plots.py \
  -f /path/to/artifact.pkl.gz -o /path/to/plots -n processing_view \
  -y 2022 --sr --variables lj0pt --binning processing
```

To request the maintained fitting view:

```bash
python analysis/topeft_run2/make_cr_and_sr_plots.py \
  -f /path/to/artifact.pkl.gz -o /path/to/plots -n fitting_view \
  -y 2022 --sr --variables lj0pt --binning fitting

python analysis/topeft_run2/make_cards.py /path/to/artifact_np.pkl.gz \
  --out-dir /path/to/cards --var-lst lj0pt \
  --ch-lst '^2lss_.*' --binning fitting
```

The plotter default is `processing`; the card default is `fitting`. State the
mode explicitly in validation commands so a default change cannot make two
comparisons ambiguous. `--rebin-plot-vars` is an additional presentation-only
integer rebin after the chosen canonical view; it does not modify `axes.py` or
establish a fitting definition.

For a plotted category combining several physical channels,
`resolve_common_axis_edges` requires all exact channel resolutions to be
identical. It does not choose the first channel's edges or coerce incompatible
views. Cards resolve one physical channel at a time.

## Understand exact aggregatability and flow

Every processing definition is either:

```python
"processing": {"kind": "uniform", "bins": N, "start": LOW, "stop": HIGH}
```

or:

```python
"processing": {"kind": "edges", "edges": [EDGE_0, EDGE_1, ...]}
```

A fitting view has a required `default` list and optional `channels` mapping.
Edges must be finite and strictly increasing. Exact aggregation requires:

1. fitting and processing share the first finite boundary;
2. the fitting final boundary does not exceed the processing final boundary;
3. every fitting boundary is exactly one processing boundary; and
4. the live histogram has one physical dense axis with underflow and overflow.

`build_aggregation_map` maps every source bin to an output bin. The original
underflow stays underflow. The original overflow stays overflow. If the fitting
view ends before the processing view, all finite source bins at or above that
final fitting boundary join output overflow. `aggregate_array` sums mapped
values; `rebin_histogram` applies the same map to every populated sparse and EFT
slot and returns a new object. There is no interpolation, partial-bin split, or
mutation of the source histogram.

Equal array lengths are not sufficient compatibility evidence. Consumers
compare physical edges and require nominal/sumw2 siblings to agree before
aggregation.

## Add a new histogram family

Adding a family begins upstream; a plotting or card list cannot create it:

1. Add the complete entry to `axes.info` with `name`, label, and one typed
   `processing` definition. For a 2D scalar family, use the existing
   `axes.info_2d` schema; its internal axes are not separate selectable
   histogram families.
2. Make `AnalysisProcessor` allocate and fill the family from an established
   physics quantity, including nominal scalar/EFT and systematic behavior at
   their current owners.
3. Decide application-axis regions and sumw2 policy/consumer requirements. A
   selected companion must use the same physical processing edges.
4. Add `fitting.default` only when a downstream view is needed. Add exact
   channel entries only for literal physical channels that differ from default.
5. Thread the family through maintained wrapper hist lists, plotting metadata,
   card distribution selection, and scaling production only where supported.
6. Update registry, processor output, nominal schema, sumw2, artifact sidecar,
   plot/card, and EFT/scaling tests for the new family.
7. Produce new PKLs: no existing artifact contains the new physical family.

At minimum run `tests/test_axis_binning.py`, a focused processor histogram
output test, `tests/test_analysis_processor_eft_sumw2.py` when companions are
supported, sidecar/merge coverage, the affected plotting test, and
`tests/test_datacard_late_rebin.py` when cards use the family.

## Add or change a fitting default

Edit `fitting.default` for the family in `axes.py`:

```python
"fitting": {
    "default": [0, 100, 200, 400],
}
```

Then:

1. validate strict increase and exact membership in processing edges;
2. enumerate physical channels using the default and those with exact
   overrides—the override remains authoritative and does not inherit the new
   default;
3. check every grouped plot category still resolves a common view;
4. apply the view to nominal and sumw2 siblings and compare their physical
   edges/content aggregation;
5. confirm every EFT coefficient slot and scaling extraction uses the same
   rebinned histogram as the template; and
6. reproduce affected plots, cards/templates, `scalings-preselect.json`, final
   `scalings.json`, and EFTFit inputs.

Existing PKLs may be reused only when their stored processing edges are exactly
compatible and artifact provenance/other contracts remain valid. A fitting
change never rewrites a source sidecar or claims the physical stored axis
changed.

## Add or change an exact channel-name override

Use the complete physical channel label as a literal dictionary key:

```python
"fitting": {
    "default": [0, 100, 200, 400],
    "channels": {
        "3l_m_offZ_low_1b_2j": [0, 200, 400],
    },
}
```

`fitting.channels` keys must be non-empty exact strings. Resolution is
`channels.get(channel, default)`. It never applies a regular expression,
substring, prefix, insertion order, or first-match rule. This is separate from
`make_cards.py --ch-lst`, which may use regex patterns to select which physical
channels are processed before exact binning resolution.

After editing an override:

1. validate the key against the authoritative channel registry;
2. test the exact channel and a similar spelling that must retain the default;
3. validate grouped-channel compatibility;
4. check exact aggregation and flow for nominal, sumw2, and EFT content; and
5. reproduce downstream products for that literal channel and any group that
   contains it.

Do not use a regex-looking key. It would be an unmatched literal, not a broad
override.

## Change processing edges

Changing `processing` changes the physical stored histogram. It always requires
new PKLs for that family. Before producing them:

1. validate every fitting default/override against the new source edges;
2. update processor axis allocation and any fixture that asserts physical
   edges;
3. ensure nominal scalar/EFT content and `<family>_sumw2` companions share the
   new processing axis;
4. review merge/sidecar compatibility so old and new physical schemas fail
   rather than concatenate;
5. validate nonprompt/other transformations that carry the family;
6. reproduce plots, cards/templates, preselected/final scalings, and EFTFit
   inputs; and
7. record the new artifact/provenance boundary in production evidence.

Changing only the fitting edges can be downstream-only when exact aggregation
from accepted existing PKLs succeeds. Changing processing edges, adding a
family, changing what observable fills a family, or requiring a previously
absent companion needs new PKLs. A downstream-only change still requires new
derived plots/cards/scalings; it means only that event processing need not be
repeated.

## Keep nominal, sumw2, EFT, and scaling content aligned

A physical family can contain scalar nominal, EFT coefficients, systematic
slots, and a sumw2 sibling. The selected processing/fitting map applies to all
content transported by the consumer:

- nominal and sumw2 must have identical stored physical edges before rebin;
- exact aggregation sums all sparse/EFT slots, not just displayed nominal
  yields;
- card templates and scaling-coefficient extraction use the same rebinned
  channel histogram;
- `scalings-preselect.json` bin order/count therefore follows the selected card
  view; and
- final `scalings.json` preserves that payload while relabeling channels.

Rebinning only the visible nominal histogram makes uncertainty bands,
autoMCstats inputs, EFT response, or scaling coefficients inconsistent. Such a
product must fail validation rather than be used downstream.

## Preserve `ptz` and `ptll` semantics

`ptz` is the transverse momentum of the in-window SFOS Z candidate used by
on-Z/genuine-Z-like categories. `ptll` is the closest-SFOS dilepton transverse
momentum used only by the explicitly mapped final three-lepton off-Z
high/low categories. Off-Z-none categories remain card-facing `lj0pt`.

An old artifact storing a value under `ptz` does not satisfy a current `ptll`
lookup. Do not alias or rename the family. Tests in
`tests/test_ptll_semantic_contract.py` protect processor routing, exact channel
membership, plot/card use, final distribution mapping, sumw2 allocation, and
scaling identity.

## Validation and failure matrix

| Contract | Validation owner |
| --- | --- |
| typed registry, exact keys, defaults, aggregation map and flow | `tests/test_axis_binning.py` |
| nominal/sumw2/EFT/scaling late-view alignment | `tests/test_datacard_late_rebin.py` |
| `ptz`/`ptll` meaning and exact category routing | `tests/test_ptll_semantic_contract.py` |
| plot grouped-channel resolution and direct defaults | `tests/test_make_cr_and_sr_plots*.py` |
| artifact physical axes and companion manifest | `tests/test_histogram_artifact_sidecars.py` |

Expected fail-closed cases include non-finite/non-increasing edges, legacy axis
keys, missing `processing` or fitting default, unknown exact override keys,
target boundaries absent from processing, a target extending past processing,
missing flow traits, more than one physical dense axis, physical edge mismatch,
and grouped channels with incompatible fitting views. Correct the definition or
reproduce compatible PKLs; do not approximate a boundary or drop a channel.

When the task also adds a category or observable, start with
[change categories or observables](categories_and_observables.md). Add the
processing axis/value/fill contract first, then add an exactly aggregatable
fitting view here. A fitting override cannot make an unfilled observable or
category current.
