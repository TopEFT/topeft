# Flexible binning

Processing binning is the physical dense axis stored in a histogram artifact.
Fitting binning is a later exact aggregation view used by cards and, when
selected, plots. Both definitions are owned by `topeft.modules.axes.info` and
resolved by `topeft.modules.axis_binning`.

## Registry schema

Each one-dimensional family has a `processing` mapping. It is either a uniform
axis with `bins`, `start`, and `stop`, or an explicit `edges` list. A `fitting`
mapping has a required `default` edge list and may have `channels`.

`fitting.channels` is a dictionary whose keys are non-empty **exact physical
channel names**. It is not a regex registry. `resolve_axis_edges(family,
mode="fitting", channel=name)` performs exact dictionary lookup and otherwise
uses `fitting.default`. If known-channel validation is requested, an unknown
key fails before use.

Every fitting edge must be present in the processing edge sequence. Exact
aggregation preserves underflow and overflow and rejects non-increasing,
non-nested, or physically incompatible axes. Nominal and `<family>_sumw2`
siblings must resolve through the same view.

## Region, distribution, and family map

| Physical category family | Final distribution | Registry owner |
| --- | --- | --- |
| Ordinary 2lss and 4l categories, plus off-Z `none` 3l categories | `lj0pt` | `axes.info["lj0pt"]` |
| Forward 2lss and forward 3l categories | `lt` | `axes.info["lt"]` |
| `2los_onZ_1tau` and selected 3l on-Z categories | `ptz` | `axes.info["ptz"]` |
| `2lss_*_1tau_onZ` categories | `ptz_wtau` | `axes.info["ptz_wtau"]` |
| Explicitly split 3l off-Z `high`/`low` categories | `ptll` | `axes.info["ptll"]` exact channel overrides |

`ptz` and `ptll` are distinct stored observables. `ptz` is the selected
in-window Z-candidate transverse momentum; `ptll` is the closest-SFOS
dilepton transverse momentum used for the explicitly mapped final off-Z
categories. There is no `ptz` fallback when `ptll` is required.

## Developer interfaces

All symbols are developer-facing and have signature authority in
`topeft.modules.axis_binning`.

| Fully qualified symbol | Parameters and return | Contract and failure boundary |
| --- | --- | --- |
| `axes.info` | Mapping from 1D family name to axis configuration | Configuration authority for physical processing edges, label, and optional fitting default/exact-channel views. Numeric values remain source-owned. |
| `axes.info_2d` | Mapping from family name to an `axes` sequence | Processing-only configuration for scalar 2D families. Its internal axis names are not independent selectable histogram families. |
| `axis_binning.processing_edges` | One family/axis mapping → tuple of numeric edges | Accepts exactly a uniform `bins`/`start`/`stop` schema or explicit `edges`; rejects missing/unknown keys, nonpositive bin count, too few edges, non-finite values, or non-increasing edges. |
| `axis_binning.validate_axis_config` | Axis mapping and optional iterable of known channels → `None` | Validates processing plus fitting schema, exact override keys, and exact aggregatability. Unknown exact keys fail when `known_channels` is supplied. |
| `axis_binning.validate_axis_registry` | Registry and optional known channels → `None` | Applies `validate_axis_config` to every family; errors identify the invalid family/configuration. |
| `axis_binning.resolve_axis_edges` | Family string; mode `processing` or `fitting` default `fitting`; optional exact channel; optional registry default `axes.info` → tuple of edges | Processing returns stored edges. Fitting uses `channels.get(channel, default)`; no regex, ordering, or first-match behavior exists. Rejects unknown family/mode or invalid registry content. |
| `axis_binning.resolve_common_axis_edges` | Family, iterable of exact channels, mode, optional registry → one edge tuple | Resolves every requested channel and requires one common result. Used where one output combines channels; incompatible views fail instead of choosing one. |
| `axis_binning.build_aggregation_map` | Source and target edge sequences → flow-inclusive source-bin-to-target-bin index array | Requires equal first finite boundary, target end not beyond source, and every target edge exactly in source. Underflow and overflow remain distinct output flow bins; finite source bins at or above a shorter target end map into output overflow. |
| `axis_binning.aggregate_array` | Array-like values, aggregation map, axis index default final → aggregated array | Sums each mapped source range along the selected axis. Invalid dimensionality/axis/map propagates as a value/index error. |
| `axis_binning.histogram_dense_edges` | Histogram → tuple of physical dense edges | Requires one dense physics axis in the supported sparse/EFT representation. Used before compatibility/rebin operations. |
| `axis_binning.validate_matching_histogram_edges` | Two histograms → `None` | Rejects different physical stored axes; consumers use it for nominal/companion and merge compatibility. |
| `axis_binning.rebin_histogram` | Histogram and target edges → new rebinned histogram | Requires one physical dense axis and exactly aggregates every populated sparse/EFT coefficient slot with flow. It does not mutate the source histogram. |
| `axis_binning.resolve_and_rebin_histogram` | Histogram, family; mode/channel/registry options → original or rebinned view | Composes resolution and rebinning. Processing mode can return the physical view; fitting mode creates the exact late view. |
| `axis_binning.make_processing_axis` | Axis config plus required name/label and optional suffix/label suffix → `hist.axis` | Constructs the production axis, including under/overflow behavior owned by the implementation. Processor construction is the principal caller. |

The plotter calls `resolve_common_axis_edges` before its late view;
`DatacardMaker.binning_view` calls `resolve_and_rebin_histogram` per physical
card channel. `AnalysisProcessor` consumes only processing axes. These functions
perform no file I/O; rebinning returns a new in-memory object.

`make_cards.py --ch-lst` is a different interface: it accepts regex selection
patterns for choosing physical channels. After selection, the chosen physical
name is passed to the exact-key fitting registry described above. Regex CLI
selection does not turn `fitting.channels` into a regex map.

To add a family, define its complete axis registry entry, make the processor
fill it, and add the relevant consumer tests. To change processing edges,
produce new PKLs because the physical stored axis changes. A fitting-only edge
or exact-channel change can reuse a compatible source PKL but requires
downstream plot/card reproduction and exact-aggregation tests.

## Source and test authority

- `topeft/modules/axes.py`
- `topeft/modules/axis_binning.py`
- `tests/test_axis_binning.py`
- `tests/test_datacard_late_rebin.py`
- `tests/test_ptll_semantic_contract.py`
