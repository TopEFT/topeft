# Datacards and scalings

## Card-production interface

`analysis/topeft_run2/make_cards.py` consumes one or more coherent histogram
artifact pairs. It accepts positional PKLs or a list file, channel/variable
selection, selected-WC controls, nuisance controls, missing-parton controls,
and local or Condor execution. The default binning view is `fitting`; year
coverage defaults to `warn`; and observed data are blinded through the default
Asimov behavior unless the supported option changes that choice.

`topeft.modules.datacard_tools.load_and_merge_histogram_pkls(...)` owns
multi-artifact compatibility and defaults to requiring sumw2. `DatacardMaker`
owns process/systematic selection, late exact aggregation, individual card and
ROOT-template contents, statistical-uncertainty policy, selected-WC discovery,
and EFT scaling extraction.

The normal WC-discovery path writes individual card/template pairs,
`selectedWCs.txt`, and `scalings-preselect.json` to the direct output directory.
With `--use-selected FILE`, `make_cards.py` reads the supplied WC selection and
materializes the same canonical signal-only representation as output-side
`selectedWCs.txt` without modifying the caller's file. A preselect record
contains a physical `channel`, a producer-owned `process`, `parameters`, and
scaling coefficient payload. Multiple records for one physical channel/process
are valid producer output.

`make_cards.build_arg_parser()` is the public option authority. Stable groups
are:

| Group | Type/default/accepted values | Contract |
| --- | --- | --- |
| Histogram input | Zero or more positional paths or one `--pkl-list-file`; exactly one source form after resolution | All inputs are sidecar-validated and merged with required sumw2. `--merge-only` stops after this gate; optional cache/report outputs preserve derived evidence. |
| Output/selection | `--out-dir .`; repeatable variables/channels; optional ignore/drop lists | `--ch-lst` patterns are regex selectors over physical channel names. This is distinct from exact fitting override keys. |
| Years/coverage | Optional supported year list; coverage `warn`, `error`, or `off`, default `warn` | Mixed input identities and structural year gaps follow the selected fail/warn policy. |
| Binning | `fitting` or `processing`, default `fitting` | Fitting performs exact late aggregation per selected physical channel. |
| WCs | POI list, reference/selected-WC files, select-only/check controls, optional WC values/scaling order | WC selection uses the card-facing fitting view; selected files must match the producer's supported process/WC structure. `--use-selected` materializes canonical signal-only `selectedWCs.txt` in `--out-dir`. |
| Nuisances | Nuisance and MC-stat switches, `--rate-syst-json`, missing-parton payload/registry/skip option | Run era selects maintained defaults when the override is omitted. An explicit `--rate-syst-json` is forwarded as `rate_systs_path`, including to Condor child commands. An explicit missing-parton path must match the selected registry/era. |
| Data/negative bins | Asimov by default; `--unblind` for observed data; crop negative bins by default | These choices affect card contents and must be recorded with production evidence. |
| Execution | Local default or `--condor`; chunks default 1 | Condor mode prepares/submits per-channel jobs and has external side effects; it requires separately authorized execution. |

`make_cards.main()` resolves inputs, validates/merges them, constructs one
`DatacardMaker`, selects or loads WCs, dispatches `run_local()` or
`run_condor()`, and writes `scalings-preselect.json`. It writes
canonical signal-only `selectedWCs.txt` for both derived and `--use-selected`
selection paths.
`run_local()` selects physical channels by regex and calls `analyze()` per
channel/distribution. `run_condor()` materializes the job boundary and
submission commands; it is a developer-facing execution interface, not a
pure-return helper.

## `topeft.modules.datacard_tools.DatacardMaker`

**Kind/status:** class, developer-facing. **Purpose:** transform one coherent
histogram family into card-facing process/systematic views, individual text
cards, ROOT templates, selected-WC information, and preselection scaling
records. **Signature authority:** class constants, `__init__`, `read`, and the
methods in `topeft/modules/datacard_tools.py`. Its `**kwargs` constructor is
weak machine-near documentation; only the curated keys below are supported.

Exactly one of `pkl_path` (path-like) or `hists` (built-in histogram dictionary)
is required. Supplying both or neither raises `ValueError`; non-dict `hists`
raises `TypeError`.

| Constructor key | Type/default | Semantics |
| --- | --- | --- |
| `binning_mode` | `fitting` default; also `processing` | Selects card-facing view. Unknown values fail. |
| `year_lst` | Sequence, default empty/all | Supported values are `UL16`, `UL16APV`, `UL17`, `UL18`, `2022`, `2022EE`, `2023`, `2023BPix`. Determines Run 2/3 rate-systematic defaults and missing-parton era. |
| `do_sm` | Boolean, default false | Selects SM-only card behavior where consumed. |
| `do_nuisance` | Boolean, default false | Loads/writes supported nuisance contracts. |
| `drop_syst` | Sequence, default empty | Removes named template systematics after supported loading/selection. |
| `skip_missing_parton_rate_syst` | Boolean, default false | Disables only the missing-parton nuisance. |
| `out_dir` | Path string, default `.` | Destination for text/ROOT card artifacts. |
| `var_lst` | Sequence, default empty | Kinematic family selection. |
| `do_mc_stat` | Boolean, default false | Enables card `autoMCStats` behavior; stored template variances still follow process policy. |
| `wcs` | Sequence, default empty | Restricts WC consideration during selection/decomposition. |
| `unblind` | Boolean, default false | Keeps real data; otherwise data processes are removed and Asimov output is used. |
| `verbose` | Boolean, default true at class level | Controls progress diagnostics. The CLI resolves its own default. |
| `use_AAC` | Boolean, default false | Enables the AAC template convention. |
| `wc_scalings` | Sequence, default empty | Requested WC ordering/selection for scaling production. |
| `rate_systs_path` | Path, default Run-2/Run-3 JSON selected from years | Rate-systematic registry relative to `topeft_path`. |
| `missing_parton_path` | Optional exact path | Overrides run-era payload selection; empty string and mixed-era use fail. |
| `sr_registry` | Registered name, default current SR registry | A nondefault registry requires an explicitly matching missing-parton payload. |
| `ignore` | Sequence, default empty | Additional pre-grouping process names to exclude. |

The constructor key is the plural `rate_systs_path`. The direct CLI forwards
that key only for an explicit `--rate-syst-json`; omission preserves the
constructor's era-selected default.

Construction reads/validates the input, loads `params/wc_ranges.json`, selects
the rate-systematic registry, optionally loads missing-parton payload content,
groups/prunes processes/systematics, and prints resolved output/read timing. It
therefore has file-read side effects even before `analyze()`. Writing occurs in
`analyze()`. Unknown leftover kwargs are not a supported extension mechanism.

Important failure boundaries include invalid/mixed years, incompatible
artifacts or WC order, missing required sumw2, unsupported SR application-axis
labels, unresolved sparse axes that would duplicate ROOT template names,
invalid exact rebinning, missing/mismatched shape pairs, and payload/registry
incompatibility.

## Scaling finalization

For `datacards_post_processing.py DATACARD_DIR -a`:

1. `ALL_CH_LST_SR` in `topeft/channels/ch_lst.json` selects the full current
   topology.
2. Source predicates map each physical category to `lj0pt`, `ptz`, `ptll`,
   `ptz_wtau`, or `lt`.
3. Physical names are sorted into `CATSELECTED`.
4. `CATSELECTED[i]` maps deterministically to `ch{i+1}`.
5. Matching cards/templates and `selectedWCs.txt` are copied to the selected
   output directory.
6. Every matching scaling record retains all producer-owned fields except
   `channel`, which becomes the corresponding `chN`; unmatched records are
   removed.
7. The selected records are written as `scalings.json`.

The historical output-directory label `ptz-lj0pt_withSys` does not imply that
all categories use only those observables. `combinedcard.txt` is neither an
input nor an output of this finalizer. EFTFit later combines the individual
cards and creates the combined card/workspace using compatible `chN` ordering.
A missing final record means no external EFT morph for that exact
channel/process.

### Finalizer CLI and artifact schemas

`datacards_post_processing.py` is public supported. Its required positional
argument is a directory already containing the individual cards/templates,
`selectedWCs.txt`, and `scalings-preselect.json`. Exactly one topology selector
is required: TOP-22 reproduction (`-s`), off-Z split (`-z`), tau (`-t`),
forward (`-f`), or current all-analysis (`-a`). `--check-condor-logs` is an
independent diagnostic flag, not a topology selector. For current TOP-26-006
production, `-a` selects `ALL_CH_LST_SR`; `-s` remains historical.

The command reads the directory and registry, creates the fixed selected-output
subdirectory, copies matching `.txt`/`.root` files plus `selectedWCs.txt`, and
writes `scalings.json`. A pre-existing output directory, missing required
inputs, or selector-count error fails. Post-copy count checks are
selector-specific: `-s` requires 43 text and 43 ROOT files, `-z` requires 75 of
each, `-t` requires 60 of each, and `-a` requires 129 of each. Every selector
requires text/ROOT symmetry; `-f` deliberately has no hard-coded exact total.
The text and ROOT diagnostics report their independent observed counts. The
finalizer returns process status rather than a library object.

`scalings-preselect.json` and `scalings.json` are JSON arrays. Each producer
record has:

- `channel`: physical `<category>_<distribution>` before finalization, then
  deterministic `chN` afterward;
- `process`: card process name, normally `<process>_sm` for EFT signals;
- `parameters`: ordered Combine-style parameter specifications beginning with
  `cSM[1]`;
- `scaling`: producer-owned per-bin coefficient payload, with underflow removed
  by `DatacardMaker`.

The finalizer changes only `channel` and filters unselected records; it must not
recompute `parameters` or `scaling`. Multiple producer records for the same
physical channel/process remain multiple records. Consumers must interpret a
missing exact channel/process record as absence of an external morph, not as a
request to borrow another process's record.

## Developer surfaces

| Fully qualified symbol | Kind/status; parameters and return | Stable contract |
| --- | --- | --- |
| `datacard_tools.process_retains_stat_uncertainty` | Developer-facing function; process string → Boolean | Only exact `fakes` or names containing `close` retain stored bin variance in ROOT output; all other templates write zero stored variance. |
| `datacard_tools.load_and_merge_histogram_pkls` | Developer-facing; paths; `require_sumw2=True`, required-family iterable empty, year policy `off` → `(histograms, merge_report)` | Validates sidecars, schema, identities, axes, disjoint contributions, policy, companion requirements, and year coverage before merge. Reads files; does not write them. |
| `datacard_tools.RateSystematic` | Developer-facing class; name and optional `all=False`, required `unc` when all | Stores process→uncertainty values. `add_process` is invalid for all-process instances; `get_process` returns the value or `-`. |
| `datacard_tools.JetScale` | Developer-facing `RateSystematic` subclass | Adds jet-indexed uncertainty lookup, symmetric low/high formatting, and a 0.01 lower bound. Missing process returns `-`; absent jet keys propagate lookup failure. |
| `datacard_tools.MissingParton` | Developer-facing `RateSystematic` subtype/marker | Holds the maintained channel-map role used while constructing the missing-parton rate systematic; payload loading/lookup is owned by `DatacardMaker` and `missing_parton_contract`. |
| `DatacardMaker.binning_view` | Histogram, family, exact physical channel → card-facing histogram | Processing returns input; fitting calls exact resolver/rebin. No regex is used here. |
| `DatacardMaker._scaling_histogram_for_json` | Channel histogram, channel, process → category-projected `HistEFT` | Internal extension. Selects nominal systematic and exact final SR application; any remaining categorical axis fails. |
| `DatacardMaker.get_selected_wcs` | Family and optional exact channel subset → process→WC sets | Inspects signal coefficient terms in fitting bins, ignoring flow, using class tolerance and optional WC restriction. No files written. |
| `DatacardMaker.make_scalings_json` | Existing record list, physical channel, family, process, WC names, scaling array → same appended list | Emits physical `<channel>_<family>`, `<process>_sm`, formatted parameters, and per-bin scaling excluding underflow. |
| `DatacardMaker.analyze` | Family, channel, selected-WC map, negative-bin policy, WC values → card result/`None` | Writes `ttx_multileptons-{channel}_{family}.txt/.root`, appends scaling records, applies fitting view to nominal and sumw2, and validates physical scaling edges. Unknown family/channel currently reports and returns `None`; deeper contract failures raise. |
| `topeft/channels/ch_lst.json` | Developer-facing configuration registry | Owns named topology blocks and physical channel/jet membership. Different consumers select explicit blocks; it is not a global claim that every block is current analysis scope. |

`DatacardMaker` extension points are the maintained process/systematic
registries, category registry, binning resolver, selected-WC policy, and
individual writer methods. Changing one requires its focused shape, sumw2,
application-axis, merge, and late-rebin tests.

See [flexible binning](flexible_binning.md) and
[missing-parton payloads](missing_parton_payloads.md).

## Source and test authority

- `analysis/topeft_run2/make_cards.py`
- `topeft/modules/datacard_tools.py`
- `analysis/topeft_run2/datacards_post_processing.py`
- `tests/test_make_cards_multi_pkl.py`
- `tests/test_datacard_late_rebin.py`
- `tests/test_datacard_tools_selective_sumw2.py`

## Physics effects and authorities

Card construction maps selected process and category content to nominal rates,
shape templates, rate/shape nuisances, MC-stat terms, and EFT scaling records.
Rate-systematic files own configured values and applicability; correction
variation labels own the candidate shape inputs; the missing-parton contract
owns its category/payload layout; sumw2 owns statistical companions; and the
binning resolver owns the exact processing-to-fitting view.

The ineffective legacy rate/envelope override is not an active default.
Current era-selected rate-systematic files and the implemented consumer remain
authority. Numeric inputs and current effects can be documented without
inferring their scientific derivation.

Category-to-distribution behavior starts from
[categories and observables](categories_and_observables.md). EFT projection
uses the shared `HistEFT` mechanism under the analysis policy described in
[HistEFT](histeft.md).

## Practical defaults and change bridge

Current numerical/applicability authorities include
[`rate_systs_run2.json`](../../topeft/params/rate_systs_run2.json),
[`rate_systs_run3.json`](../../topeft/params/rate_systs_run3.json), the
[`missing_parton_contract.py`](../../topeft/modules/missing_parton_contract.py)
registry/payload layout, and the fitting views in
[`axis_binning.py`](../../topeft/modules/axis_binning.py). The selected-WC and
scaling record are category/process dependent, so one representative record is
not a universal default.

A concrete chain is physical category + distribution → fitting view → nominal
and systematic card templates → selected WCs → physical-channel scaling
record. Use
[create cards and finalize EFT scalings](../how_to/datacards_and_scalings.md)
to change rate files, applicability, binning, selected WCs, or export. Use the
[missing-parton payload guide](../how_to/missing_parton_payloads.md) for that
separately owned payload family.
