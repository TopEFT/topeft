# Why processing and fitting binning are separate

The histogram axis stored in a PKL and the binning used by a final fit serve
different purposes. Fine processing bins retain information for validation and
alternative downstream views. A fitting view may aggregate those bins to keep
templates statistically and numerically useful.

## One physical axis, multiple exact views

`axes.info[family]["processing"]` defines the physical dense axis written by
the processor. Optional fitting metadata supplies a family default and exact
physical channel-name overrides. An override key is a literal dictionary key,
not a regular expression. The resolver returns the exact channel entry when it
exists and otherwise uses the family default.

Downstream aggregation is permitted only when every fitting edge is already a
processing edge. The aggregation map includes physical bins and preserves the
underflow and overflow treatment. It is not an interpolation and must not
invent a boundary that was absent from the stored axis.

The exactness rule protects the statistical meaning of the source artifact.
If a proposed fitting boundary cuts through a stored processing bin, the
downstream code has no information about how content is distributed inside
that bin. Refusing that view is more honest than approximating it.

## Definition and resolution model

Each one-dimensional family in `axes.info` has a processing definition. That
definition can use explicit edges or the supported uniform-axis schema. A
family that supports a fitting view adds:

- `fitting.default`, the ordinary downstream view; and
- `fitting.channels`, a mapping from an exact physical channel name to an
  alternate edge list.

`validate_axis_config` checks increasing edges, known exact channel keys, and
aggregatability. `resolve_axis_edges` performs an exact dictionary lookup for
the supplied channel and falls back to the default only when no exact key is
present. It never interprets a key as a regular expression and never chooses a
“first matching” override.

This differs from analyst-facing channel selection in `make_cards.py
--ch-lst`, which may use regular expressions to choose channels. Selection
answers “which channels?”; fitting resolution answers “which exact edges for
this already selected physical channel?”

## Exact aggregation and flow

`build_aggregation_map` maps every requested physical interval to a consecutive
set of processing bins. `aggregate_array` applies that map to arrays, while
`rebin_histogram` applies it across the histogram's sparse/EFT slots. Underflow
and overflow are carried according to the inclusive flow convention rather
than being mistaken for ordinary edge bins.

The code first identifies the semantic dense axis and checks its physical
edges. It does not infer the axis from a raw array position after slicing or
grouping, where singleton categorical axes may still be present.

## Ownership and propagation

The axis registry owns definitions. `axis_binning.py` owns validation,
resolution, and exact aggregation. Plotting defaults to the processing view;
cards default to the fitting view. Both use the same resolver so their meaning
cannot drift through duplicated bin tables.

Plot configuration may request a view, but it must not copy numeric edges into
the plotting metadata. Likewise, card configuration selects `processing` or
`fitting`; `DatacardMaker` delegates numeric resolution to the same registry.
The campaign-specific matrix wrapper is not a binning authority.

## Related histogram content

A nominal family may have scalar, EFT, systematic, and sumw2 siblings. Exact
aggregation must apply one resolved physical map to every sibling that a
consumer transports. Scaling coefficients are also extracted from the
selected fitting view. Rebinning only the visible nominal histogram would make
statistical uncertainties, EFT response, or `scalings-preselect.json`
incompatible with the card template.

Artifact sidecars record physical-axis and family evidence used to reject
incompatible fragments before late aggregation. A fitting-view change does not
rewrite the source sidecar or claim that the stored processing axis changed;
it changes the downstream product and its validation context.

## Change boundaries

Adding a new histogram family starts with a processing definition and the
processor fill that makes it real. Fitting metadata can then describe exact
coarser views. Adding a family name only to a plot or card list cannot create
the missing source histogram.

Changing a fitting default affects every channel without an exact override.
Changing one exact channel entry affects only that literal channel name.
Changing processing edges affects the source artifact for every consumer.
Because these scopes differ, review and validation should name the exact
family, view, and affected channels rather than say only “the binning changed.”

A processing-edge change requires new PKLs because it changes the stored
physical object. A fitting-edge change can be applied to compatible existing
PKLs, but plots, cards, templates, and EFT scaling payloads derived from the old
fit view must be reproduced and validated. Nominal, sumw2, EFT, and scaling
content must use the same resolved mapping.

The focused validation owners check registry schemas, exact-name behavior,
aggregation maps and flow, physical histogram edges, plot/card late rebinning,
sumw2 sibling handling, and the `ptz`/`ptll` semantic partition. Passing a
Markdown link check cannot substitute for those tests or for regenerating the
affected downstream artifacts.

`ptz` and `ptll` remain different observables. Only the explicitly mapped final
off-Z categories use `ptll`; a missing `ptll` family never falls back to
`ptz`.

See the [flexible-binning how-to](../how_to/flexible_binning.md) for changes and
validation and the [reference](../reference/flexible_binning.md) for the exact
registry and function contracts.
