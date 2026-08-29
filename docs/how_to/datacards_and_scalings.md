# Create cards and finalize EFT scalings

`analysis/topeft_run2/make_cards.py` is the maintained direct card-production
interface. It consumes one or more compatible histogram PKLs—normally the
nonprompt-transformed products—and writes individual text cards, ROOT template
files, a generated `selectedWCs.txt` unless `--use-selected` is active, and
`scalings-preselect.json`. There is no maintained general card wrapper; tracked
matrix scripts can be campaign/operator records without becoming a second
interface authority.

| Interface | Owns | Defaults/derived state | Delegates or does not own |
| --- | --- | --- | --- |
| `make_cards.py` | input merge validation, WC selection, channel/variable selection, `DatacardMaker` construction, local or generated Condor execution | fitting binning, year coverage `warn`, Asimov data, no nuisances or MC-stat opt-in | `DatacardMaker` owns card/template/scaling construction; registries, rate payloads, binning and histogram artifacts remain external authorities |
| `datacards_post_processing.py` | one topology selection, deterministic physical-channel ordering, file selection/copy, `chN` relabeling | exact one-of selector; `-a` chooses `ALL_CH_LST_SR`; destination is fixed to `ptz-lj0pt_withSys` | does not make individual cards, fit them, combine them, or recalculate producer scaling payloads |
| EFTFit/Combine | individual-card combination and statistical fit | external workflow | creates `combinedcard.txt` later; does not redefine topeft's channel/topology selection |

The region -> distribution -> binning mapping is shared source authority:
physical regions and jet populations come from `topeft/channels/ch_lst.json`;
the finalizer chooses each region's card distribution (`lj0pt`, `ptz`, `ptll`,
`ptz_wtau`, or `lt`); `topeft/modules/axes.py` supplies processing/fitting
edges. An operator matrix may record a campaign selection but must not become a
second copy of that map.

## Create individual cards and templates

From `analysis/topeft_run2`:

```bash
python make_cards.py /path/to/final_np.pkl.gz \
  --out-dir /absolute/path/to/cards \
  --var-lst lj0pt ptz ptll ptz_wtau lt \
  --ch-lst '^2lss_.*' '^3l_.*' '^4l_.*' \
  --binning fitting --year-coverage-policy error
```

Repeat positional PKLs for coherent fragments, or use `--pkl-list-file` for a
long list; do not provide both. The default uses Asimov data, fitting binning,
no nuisance insertion, and warning-only year coverage. `--unblind`,
`--do-nuisance`, `--do-mc-stat`, and `--keep-negative-bins` are explicit
choices. `--merge-only` stops after load/merge validation, and
`--merge-report PATH` retains the diagnostic report.

Before card production, confirm that every input belongs to one compatible
production family and has the nominal/sumw2 companions required by the selected
card channels and variables. A filename or campaign directory is not
compatibility evidence.

`make_cards.py` validates and merges all PKLs before constructing
`DatacardMaker`. It rejects positional inputs combined with `--pkl-list-file`,
mixed incompatible schemas/policies, and missing required companions.
`--merge-only` is the lowest-cost way to exercise that boundary. A cached
merged PKL written with `--cache-merged-pkl` is a new artifact and sidecar when
the input schema supports it; preserve its lineage and merge report.

Channel arguments are selection patterns interpreted by the existing
`regex_match` helper against actual histogram channel labels. They do not add a
channel to the registry or repair an absent artifact category. Variables must
exist as histogram families in the merged input.

## Modify card selection or configuration

- Select variables and channels through `--var-lst` and `--ch-lst`; keep
  physical channel definitions in `topeft/channels/ch_lst.json`.
- Choose processing or fitting edges through `--binning`; change definitions at
  `topeft/modules/axes.py`, following the [binning guide](flexible_binning.md).
- Use `--miss-parton-file` and `--sr-registry` to select existing supported
  configuration. Do not duplicate payload or registry data in an operator
  wrapper.
- Do not rely on `--rate-syst-json` in the current implementation. The parser
  accepts it, but `make_cards.py` passes the value as `rate_syst_path` while
  `DatacardMaker` consumes `rate_systs_path`; the supplied path is therefore
  ignored and the maker selects its Run 2 or Run 3 default rate-systematics
  file. Correcting that keyword boundary is an executable change.
- `--use-selected FILE` reads the reviewed JSON for card construction, but it
  does not copy `FILE` to `<out_dir>/selectedWCs.txt`. Before finalization,
  place and independently verify the exact reviewed `selectedWCs.txt` in the
  card output directory; otherwise the finalizer has no file to copy.
- When extending the CLI, update parsing, Condor forwarding if applicable,
  `DatacardMaker` construction, output/provenance behavior, and focused tests.

The `--condor` implementation is not transparent forwarding. It generates a
worker script and submit files with a curated option list, one CPU, 20000 MB
memory, 4096 MB disk, transferred `make_cards.py`/`selectedWCs.txt`/PKL list,
and a shared output-directory assumption. When a supported card option must
work under Condor, add it to `_build_condor_base_other_opts` and test both local
and generated-worker commands. Do not assume a new parser option reaches jobs.

To add a supported selection/configuration control:

1. Identify its existing owner: physical channels in `ch_lst.json`, axes in
   `axes.py`, currently default-selected rate-systematic JSON in
   `DatacardMaker`, missing-parton payload/registry through their dedicated
   options, or WC selection through selected-WC inputs. Treat the ineffective
   `--rate-syst-json` keyword path as a source defect, not an extension model.
2. Add a CLI selector only when choosing among existing supported authorities;
   do not copy the configuration into `make_cards.py`.
3. Validate choices before output creation and thread the resolved value to
   `DatacardMaker` once.
4. Preserve local/Condor equivalence or explicitly fail a mode that cannot
   represent the option.
5. Update `tests/test_make_cards_multi_pkl.py`, the focused option/physics
   contract test, `tests/test_split_datacard_boundary.py`, and late-rebin or
   selective-sumw2 tests when those surfaces are affected.

Card changes can affect template shapes, nuisance content, WC selection, the
preselected scaling records, and every later EFT fit. Validate the card/template
pair together rather than checking the text card alone.

The normal generated-selection output set contains one text-card/ROOT-template
pair per selected physical channel and distribution, `selectedWCs.txt`, and
`scalings-preselect.json`. With `--use-selected`, the operator must supply the
reviewed `selectedWCs.txt` in the output directory separately. The preselect
file records producer-owned EFT polynomial payloads under physical
channel/distribution labels. `make_cards.py` does not assign final `chN` labels
or create `combinedcard.txt`.

The tracked `run_make_cards_run3_yawen_matrix.sh` is classified as an archival
operator record: its site/user paths, campaign inputs/hashes, environment and
branch assumptions, and DATACARD023-qualified provenance make it useful for
that campaign but not a supported wrapper. It remains tracked in its runnable
location; moving, removing, or generalizing it requires a separate executable
source-control decision. Use `make_cards.py` directly for maintained work.

## Finalize the current full topology

The card directory must already contain the individual card/template pairs,
`selectedWCs.txt`, and `scalings-preselect.json`. Then run:

```bash
python datacards_post_processing.py /absolute/path/to/cards -a
```

Exactly one topology selector is required. `-a`/`--all-analysis` selects the
full current topology from `ch_lst.json`. The script deterministically orders
the physical channel names, maps them to `ch1`, `ch2`, and so on, copies the
selected card/template/WC inputs, and writes final `scalings.json`. A matching
scaling record retains its producer-owned payload while only its channel label
is replaced by the deterministic `chN` label.

Use a card directory whose `ptz-lj0pt_withSys` destination does not already
exist; the finalizer creates it and does not implement resume/overwrite. It
has incomplete selector-dependent count guards: `-s`, `-z`, and `-t` check both
text and ROOT totals; `-a` checks only the ROOT total; and `-f` checks neither.
The printed "root templates copied" line also reports the text counter. Do not
use that line or a successful exit as completeness evidence. Independently
list and count the copied `.txt` and `.root` files, pair them by the expected
physical `<channel>_<distribution>` stem, and compare the pairs with the
selected topology before handing the directory onward.

The mapping procedure is deterministic: load the chosen registry block,
expand its physical jet populations, choose the source-owned distribution for
each category, sort the resulting physical channel/distribution names, and map
their one-based positions to `ch1`, `ch2`, .... Only scaling records whose
physical label is selected are retained. A record's coefficients/WC order are
not recalculated during relabeling.

`combinedcard.txt` is neither an input nor an output of
`datacards_post_processing.py`. EFTFit later combines the individual cards and
creates `combinedcard.txt` before the Combine handoff. If the final scaling
file has no record for an exact channel/process pair, that pair has no external
EFT morph; do not fabricate one during finalization.

Validate this boundary with `tests/test_split_datacard_boundary.py`,
`tests/test_ptll_semantic_contract.py`,
`tests/test_datacard_late_rebin.py`, and the relevant card-option tests. A
binning change can alter template bin counts and `scalings-preselect.json`, so
reproduce cards and final scalings together even when existing PKLs remain
exactly aggregatable.

## Diagnose card/finalization failures

| Failure | Correct response |
| --- | --- |
| merge policy/schema/companion mismatch | fix or reproduce the upstream artifact; do not concatenate dictionaries manually |
| selected variable/channel absent | inspect the merged histogram axes and source registry; a regex cannot create missing content |
| fitting edges not exactly representable | correct the canonical fitting view or produce compatible processing-binned PKLs |
| selected-WC mismatch | review the new selection and reference; do not skip the check without an explicit validated reason |
| finalizer missing card/template | reproduce that physical channel/distribution pair; do not let relabeling hide an incomplete set |
| preselect scaling has no selected physical label | determine whether the process intentionally has no external EFT morph or the producer output is incomplete |
| destination already exists | choose a fresh card-finalization directory; there is no supported resume/merge behavior |

The selectors `-s`, `-z`, `-t`, and `-f` describe narrower or historical
topologies. In particular, `-s` is the historical TOP-22-006 selection, not the
current default. See the [historical TOP-22-006 page](historical/top_22_006.md).

Exact schemas and option contracts are in the
[datacards/scalings reference](../reference/datacards_and_scalings.md). The
[EFTFit boundary explanation](../explanation/datacards_and_eftfit.md) describes
why finalization and card combination remain separate responsibilities.

Use [categories and observables](categories_and_observables.md) before changing
the physical category or distribution that a card consumes, and use
[corrections, weights, and systematics](corrections_weights_and_systematics.md)
before changing the upstream variation. This guide owns the card-facing rate,
applicability, fitting-view, selected-WC, and scaling-export changes only.
