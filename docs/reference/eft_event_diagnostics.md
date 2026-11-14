# EFT event diagnostics

## Manifest campaigns

`analysis.topeft_run2.event_scan_condor` owns multi-unit orchestration around
its existing per-unit `prepare`, `submit` and the scheduler's `observe` protocol.
From the repository, using the installation wrapper and interpreter:

```text
python -m analysis.topeft_run2.event_scan_condor plan-manifest --manifest MANIFEST
python -m analysis.topeft_run2.event_scan_condor prepare-manifest \
  --manifest MANIFEST --campaign-root OWNER_SELECTED_ABSENT_ROOT \
  --wrapper WRAPPER --python PYTHON --request-memory-mb 8192 --max-runtime 3600
python -m analysis.topeft_run2.event_scan_condor submit-manifest --campaign-root OWNER_SELECTED_ROOT
python -m analysis.topeft_run2.event_scan_condor status-manifest --campaign-root OWNER_SELECTED_ROOT
```

The campaign root is mandatory, has no default, and is selected by the caller
for operational storage. Diagnostic handoff wrappers are historical evidence;
launch through the maintained module. `plan-manifest` only reads manifest bytes,
checks structural unit identities/ranges, and returns sorted units and the exact
SHA-256. It does not validate scientific inputs, read ROOT files, prepare outputs,
or contact the scheduler.

`prepare-manifest` freezes the exact manifest bytes and prepares units in sorted
unit-ID order using canonical per-unit descriptors. Version-1 `campaign.json`
binds the original manifest path/hash, root, ordered unit IDs, runtime and resource
bounds with a content token. It is published last. Each `units/UNIT_ID` owns its
attempt metadata, intents/responses, scheduler identity, worker receipts and
`science` outputs. Preparation re-entry accepts only a fully prepared, exactly
matching campaign; partial preparation and unknown collisions fail closed and
remain for inspection. Moving the campaign, changing runtime/resource choices,
or changing/removing either original or frozen manifest invalidates re-entry.

`submit-manifest` locks the campaign and checks every unit before new effects.
A durable exclusive `campaign_submit_intent.json` precedes each canonical
per-unit `submit` call. Any existing partial attempt blocks the campaign, even
if interruption happened before scheduler contact. No automatic retry or intent
removal is supported. Consistent prior submissions are skipped regardless of
worker success/failure; skipping never asserts completion. After interruption,
re-entry can continue only provably never-attempted units, provided all earlier
attempts have consistent identities/responses. Contradictions block new effects.
All controllers require the same filesystem's working exclusive flock protocol;
direct per-unit operations must not run concurrently with campaign control.

`status-manifest` requires separately authorized scheduler access. It observes
each canonical descriptor with `read_only=True`, returns individual scheduler
states/ads plus worker receipts and local ambiguity, and never submits, retries,
or persists recovered identities. Save its JSON stdout as a campaign readback
when authorized. Scheduler terminal success alone does not establish scientific
coverage. Synthetic tests qualify orchestration; production execution and
scientific closure require separate evidence and authorization.

## Single-source scans and Condor execution

`analysis.topeft_run2.event_scan` composes the existing native fill observer for
one physical ROOT source and one explicit entry range. Run every command through
the wrapper and interpreter resolved by the installation profile, from the TOPEFT
repository. The direct interface is:

```text
python -m analysis.topeft_run2.event_scan scan \
  --source-root SOURCE.root --sample-json SAMPLE.json \
  --analysis-config SAMPLE.cfg --target-manifest TARGETS.json \
  --entry-start START --entry-stop STOP --output-dir ABSENT_DIRECTORY
```

Start is inclusive, defaults to zero and must be an integer >= 0. Stop is
exclusive and optional: omission means the tree end. Empty, reversed, negative
and beyond-tree ranges fail before scientific output. The sample CFG is the
hash-bound sample configuration; the SR registry is separately bound in TARGETS.
There are no filename-dependent resume rules. An output directory must be absent;
there is no overwrite, automatic retry or resume operation.

The version-1 target JSON contains `group` (group_id, dataset, native process,
metadata_year, year_or_era, tree_name), `ownership` (the existing native observer
rows, including term, native_term_index, source_channel, fit_variable, source_bin,
source_edges and negative_row_id), `sr_config_path`, `calibrations` (path/SHA-256
pairs), and `input_sha256`. Bind sample JSON, sample CFG, SR registry, processor,
diagnostic helper, scanner and all scientific dependencies/calibrations actually
used. The worker verifies native WC order, selects the bound SR categories and
uses the existing `observed_processor`, `fill_capture` and HistEFT product logic.

`event_scan.make_unit(source, sample, cfg, targets, entry_start=0, entry_stop=None)`
performs metadata-only source binding and returns an immutable unit with source
path/size/mtime, ROOT UUID, tree entries, target hash and range. Device/inode check
local replacement during each header read; they are not cross-node identifiers. A
version-1 execution manifest is `{"schema_version":1,"units":[UNIT,...]}`. Run one
unit with `event_scan run --manifest FILE --unit-id ID --output-dir ABSENT_DIR`.
Both interfaces invoke the same range worker and serialize the same science.

Each result contains the native `events.tsv.gz`, signed per-target partial
`sums.json`, `runtime.log` and `receipt.json`. Native event fields retain identities,
ordinary weight, constant/diagonal coefficients, products and dtypes. Receipts
bind source/config/code, requested and processed ranges, completed chunks, target
identities, calibration opens and artifact hashes. A completed strict subrange
has `range_complete=true`, `whole_file_complete=false`; interrupted chunks never
complete coverage. Default bounds are 10,000 entries/chunk, 3,600 seconds, 256 MiB
event output and a 1 GiB free-space reserve. No intra-job parallelism is added.

`event_scan coverage --receipt RECEIPT ...` revalidates source identity and
artifact hashes and requires contiguous coverage of the whole tree. Optional
`--entry-start/--entry-stop` restrict the required interval. It rejects failed
receipts, incompatible target/code contracts, duplicate/overlapping ranges and
unexplained gaps. Source identity must be revalidated before reusing a receipt.
Zero selected rows prove only the completed requested range; a period zero also
requires the independently known complete source set.

`analysis.topeft_run2.event_scan_condor prepare --manifest FILE --unit-id ID
--output-dir ABSENT_ATTEMPT --wrapper WRAPPER --python INTERPRETER` prepares one
unit without querying or submitting. `submit --execution ATTEMPT/_metadata/execution.json`
submits that unit at most once; `status --execution ...` observes it without
resubmission. Default resources are one CPU, 8 GiB memory, 1 GiB disk and 3,600
seconds; preparation exposes memory/runtime options. One ROOT file is the normal
unit; exceptional or previously partial sources use ordinary nonoverlapping ranges.

The adapter reuses `plot_campaign_condor` target resolution, durable intent,
submit/identity parsing and terminal reconciliation, plus durable JSON helpers.
Its inherited `PlotExecutionToken` ClassAd attribute is an execution identity,
not a plot request. Its separate worker is necessary because the existing plot
worker accepts only the renderer. The event worker runs through the configured
wrapper, verifies descriptor/code/input identity and persists an exclusive start
marker before invoking the same scanner. Eviction/restart cannot start a second
scientific attempt. A failed or ambiguous submit requires bounded queue/history
readback and stops; this adapter exposes no retry or recovery submission API.
Scheduler terminal success and scientific receipt qualification remain separate.
Commands and manifests never grant execution authority or imply campaign closure.

`analysis/topeft_run2/eft_event_diagnostics.py` provides one local diagnostic CLI
for nominal EFT event contributions. It observes the maintained
`AnalysisProcessor` immediately before `HistEFT.fill`; it does not reconstruct
selection or correction logic. The observed module is an in-memory copy with
an exact source hash, two native `PackedSelection` constructors composed into
shards, one callback, and suppression of terminal histogram accumulation.
The processor file and calibration payloads are never modified. This retains
the event-observation and SM sumw2 methodology established by MP009A3 and the
R6C1/R6C2/R6C4 diagnostic evidence.

Run from the repository with the configured environment and command wrapper.
The existing scientific libraries are required; no installation is performed.

```text
python -m analysis.topeft_run2.eft_event_diagnostics --help
```

The five operations are:

The in-memory `combined_producer_views(components, family, years, channels)`
and `combined_bin_coefficients(bundle, channel, process, period=None)` APIs
also expose aggregate and source-period coefficients through the maintained
DatacardMaker selection and fitting binning. They do not open analysis PKLs.
Callers bind exact source-unit/channel/process/physical-bin/WC-order tuples
and deserialize only their separately authorized inputs.

Scalar and EFT origin tags retain the complete combined categorical order and
sparse support, including zero-valued coordinates. Native process grouping
therefore sees the same process universe for both tags. In Run3 ttll, the
combined producer retains the direct private ttll process rather than the
scalar samples mapped to the same group name. Independent scalar-only and
EFT-only makers have different ownership and cannot define physical S/E0.
The tags close coefficient by coefficient against the combined result.
S is scalar coefficient zero, E0 is EFT coefficient zero, A0 is combined
coefficient zero, and Eii uses `diagonal_index` in the returned native WC order.
Eii/E0 is defined only for exact nonzero E0; Eii/A0 is the package scaling.

The returned physical-bin arrays remove source underflow once and retain the
native overflow as the final physical bin, matching the maintained scaling
writer. Missing combined process/period support raises; origin absence is zero
only relative to a present combined process. Signed A0 is preserved even when
a package nominal template is sanitized or absent, as in the ch235 evidence.
ROOT nominal contents and TXT process-integral rates remain separate package
readbacks; neither is silently substituted for signed A0.
`package_bin_representation` describes equality, absence, or a negative signed
denominator with a zero template; the latter two require exact package lineage
before scientific acceptance. An I/O failure must never be passed as absence.

| Operation | Inputs and output |
| --- | --- |
| `discover` | Accepted year/era TSV, prior ownership/context/group evidence, source binding, calibration read scope, exact exclusion channel/variable; emits the retained and excluded worklists and immutable manifest. |
| `event-closure` | Manifest and absent output paths; reads local ROOT sources once per ownership context in sequential entry ranges, emits event factors, per-target closure, mechanisms, unresolved targets and an execution receipt. |
| `weight-decompose` | Serialized R6C2-format component table; independently multiplies finite named factors. |
| `btag-decompose` | Serialized per-jet Method-1a table; reconstructs medium, loose-not-medium and not-loose factors, subgroup products and the native zero-denominator guard. |
| `readback` | Frozen R6C1/R6C2/R6C4 evidence directory; checks event products, SM sumw2, per-era sums, preserved ordinary factors and the negative loose-not-medium fixture. |

`btag-decompose` reconstructs the serialized raw Method1a factors, including
`eL * SFL - eM * SFM` for Loose-not-Medium and the zero-MC guard.
The frozen `readback` evidence retains that historical negative interval.
Neither operation applies the selected `topcoffea` helper's bounded
negative-data interval fallback, so these reconstructions do not establish
parity with the current runtime b-tag weight. For current processor use, see
[the Method1a consumer](../how_to/corrections_weights_and_systematics.md#understand-the-b-tag-method1a-consumer).

`discover` consumes the accepted diagnostic ownership schema (negative row IDs,
source groups, native bins and WC names). It is not a generic ROOT analyzer.
The prior filenames are part of this version's explicit input interface.
Accepted target values are copied literally, never recomputed during discovery.
The exclusion must equal the complete serialized exclusion identity set. Every
retained term is mapped through actual `WCnames` and the maintained packed EFT
helper. Duplicate ownership and changed source/config/input identities fail.

The manifest freezes numerical tolerances before ROOT reads. Defaults reuse
the prior event-observer convention `atol=1e-10`, `rtol=1e-10`; negative targets
additionally require a strictly negative observed sum. A zero reconstruction
cannot close a tiny negative target just because it lies below the absolute
tolerance. Tolerances are diagnostic agreement criteria, not significance tests.

## Event evidence and coverage

`class_b_event_contributions.tsv.gz` has one row per selected physical fill.
`fill_id` comprises source context, entry, channel, variable and native source
bin. Run/lumi/event and the absolute source filename are retained. Multiple
variables may refer to the same physical event; the ROOT entry is not rescanned
per term or variable. Only requested diagonal coefficients are serialized.

The `indices`, `pii`, `qii`, `sign_pii` and `sign_qii` columns contain aligned JSON
arrays; `indices` uses the sample's native packed order. Each row also records
`w_analysis`, `p0`, `q0`, native dtypes and `sign_w`. Products follow HistEFT's
flatten/repeat/multiply dtype behavior. Negative ordinary weights additionally
record every stored nominal factor from `Weights.partial_weight`; private
`_weights` keys are used only to enumerate factors, an explicit API dependency.
No BTV prevalence follows from a negative weight alone.

Ownership links each fill to all relevant term/era targets. Signed sums use
bounded-memory compensated accumulation. The receipt records every started and
completed context and every completed entry range. Only completed contexts enter
the summary sums; rows from an interrupted context remain preserved, and its
targets remain unresolved. A target closes only after all owning groups finish.
The first complete-group closure mismatch stops further ROOT reads.

Mechanism labels describe the source of **negative event products** contributing
to a closed negative target. Negative weight with positive diagonal coefficient
gives `ORDINARY_WEIGHT_ORIGIN`; positive weight with negative coefficient gives
`EFT_COEFFICIENT_ORIGIN`; occurrence of both gives `MIXED`. Both factors negative
produce a positive contribution, retained in sign counts but not treated as a
negative-product cause. Incomplete or nonclosing targets are `UNRESOLVED`.

## Execution and failure contract

All output paths must be absent. There is no overwrite, retry or resume mode.
The caller supplies an authorized diagnostics directory; invocation itself does
not grant filesystem or scientific authority. The default limits are one ROOT
source, 10,000 entries per chunk, 30 hours, 2 GiB event output and a 750 MiB free
space reserve. Capacity is checked before each chunk. Limits may stop an
exhaustive attempt; partial coverage is never reported as complete.

A Python audit hook denies network operations and every pickle open except the
exact read-only native calibration allowlist in the owner-authorized manifest.
Analysis/reference pickles and `_np` inputs are never used. Calibration identity
and actual open counts are separate from analysis-pickle accounting. No scheduler
or remote-Git command is implemented. On an execution error the CLI closes the
compressed stream, preserves progress, emits all target statuses and returns
nonzero. Prelaunch identity, import or collision errors occur before the scan.

The frozen R6C2 artifact preserves 19 factors and was originally marked partial;
its regression checks that serialized product without fabricating missing nominal
unit-factor rows or claiming complete historical component coverage. R6C4 supplies
the separately serialized BTV decomposition.

For an independent scientific closure claim, a separate readback must reopen the
manifest, event table and receipt and recompute every target from serialized
factors and ownership, without importing or executing this collector. The CLI's
own target summary is provisional evidence. A small, one-off standard-library
readback invocation can perform that check; it is not another maintained event
launcher. It must apply the same predeclared tolerances, verify native product
rounding, complete context membership, unique fills and sign counts, and preserve
all unresolved targets. Passing unit or frozen tests is not full-population closure.
