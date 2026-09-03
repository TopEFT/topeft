# Run and extend CR/SR plotting

Use `run_plotter.sh` for the maintained convenience interface and
`make_cr_and_sr_plots.py` when you need the complete plotting CLI. Both consume
existing histogram PKLs; they do not run the processor or mutate the input
artifact.

| Interface | Owns | Delegates or derives | Default source | Downstream outputs |
| --- | --- | --- | --- | --- |
| `run_plotter.sh` | shell-level required-path/year checks, alias expansion, filename-based region fallback, wrapper option translation, dry run | delegates rendering to `make_cr_and_sr_plots.py`; derives normalized years and CR/SR flag | shell locals: quiet output, one worker; direct CLI supplies unset defaults | same figures/reports as direct CLI |
| `make_cr_and_sr_plots.py` | artifact merge/coverage checks, region context, process grouping, blinding, systematics, binning view, rendering and reports | reads plotting metadata and canonical axis definitions | argparse: output `.`, name `plots`, merged channels, processing binning, one worker, year coverage `warn`, negative report enabled | plot directories, HTML indexes, optional zero-yield report, negative-contribution CSV/Markdown |

The wrapper forwards unrecognized tokens, but forwarding does not make shell
code the semantic owner of the option. Exact direct options stay in the Python
parser and [plotting reference](../reference/plotting.md).

## Plot with `run_plotter.sh`

Run from `analysis/topeft_run2`:

```bash
./run_plotter.sh -f /path/to/final_np.pkl.gz \
  -o /absolute/path/to/plots -y run3 --sr \
  --variables lj0pt ptz ptll lt --channel-output merged --dry-run
```

The wrapper validates paths and year tokens, expands `run2`/`run3`, resolves CR
or SR from an explicit flag or filename, and forwards plotting options. It does
not own channel maps, process grouping, styles, or binning definitions. Keep
`--dry-run` until the resolved command and output destination are correct.
`run_plotter.sh` creates the requested output directory before it reaches the
dry-run guard, so `--dry-run` is not a filesystem no-op. Use a deliberate fresh
or disposable output path when inspecting the command.

An explicit `--cr` or `--sr` is preferable for production plots. Filename
detection is a convenience: if both `CR` and `SR` appear, the wrapper warns and
uses its control-region fallback unless explicitly overridden. `--blind` and
`--unblind` override the direct plotter's region-dependent data choice.
`--channel-output` accepts `merged`, `split`, `both`, and their `-njets`
variants; the latter retain per-jet bins defined by plotting metadata.

When extending the wrapper, prefer transparent forwarding. A wrapper-owned
option should exist only when the shell layer must derive or validate a value
before calling Python. Preserve the contracts in
`tests/test_make_cr_and_sr_plots.py` and focused wrapper/Condor tests.

For a wrapper-owned option, update help, value validation, conflict handling,
normalized command construction, and dry-run output. For a forwarded option,
leave semantic validation in Python and test that token grouping survives the
shell boundary. Do not copy a Python default into the wrapper unless the
wrapper deliberately changes it. Check sourceability/return handling and the
Condor forwarding path when the new option is relevant there.

## Plot directly

```bash
python make_cr_and_sr_plots.py \
  -f /path/to/final_np.pkl.gz -o /absolute/path/to/plots -n run3_sr \
  -y 2022 2022EE 2023 2023BPix --sr \
  --variables lj0pt ptz ptll lt --binning fitting \
  --year-coverage-policy error
```

Repeat `-f` for coherent fragments or use `--pkl-list-file`. Direct plotting
defaults to the processing-binning view, merged channel output, one worker,
warning-only year coverage, and enabled negative-weight reports. A mixed Run 2
and Run 3 year selection is rejected.

The plotter validates and merges repeated inputs before rendering. Choose
`--year-coverage-policy error` for a production validation pass; `warn` records
structural gaps but continues, and `off` suppresses that check. More than one
worker parallelizes variables and then categories when slots remain, but each
process loads the histogram dictionary, so memory use grows with worker count.

The negative-contribution report is diagnostic, not a correction. It records
signed process/group yields, sumw2-derived errors/effective entries when a
companion exists, and total-MC context. `--no-negative-weight-report` disables
only that report. `--rebin-plot-vars` is a presentation/report-time integer
rebin after the selected processing or fitting view; it is not a new canonical
axis definition.

## Change plotting configuration at its owner

`topeft/params/cr_sr_plots_metadata.yml` owns the maintained plotting metadata:

- `CR_CHAN_DICT` and `SR_CHAN_DICT` map displayed category names to histogram
  channel labels;
- `CR_GRP_MAP` and `SR_GRP_MAP` group raw processes for stacked plots;
- `REGION_PLOTTING` owns region transformations, removals, opt-in skips, and
  blinding mechanics;
- `STACKED_RATIO_STYLE` owns figure, axis, tick, and legend presentation.

Do not copy those values into `run_plotter.sh`. A metadata change can affect
which channels/processes are shown, how yields are combined, blinding, and all
produced figures. Update the plotting tests covering the changed block and
validate at least one representative CR or SR configuration statically or with
the permitted lightweight test scope.

When adding or changing a plotting category:

1. Confirm the physical channel labels present in the producer artifact and the
   intended merged/split presentation.
2. Edit `CR_CHAN_DICT` or `SR_CHAN_DICT` and the appropriate group/region block
   in `cr_sr_plots_metadata.yml`; avoid an equivalent hard-coded Python or shell
   list.
3. Decide whether the change is selection, grouping, transformation, skip,
   blinding, or styling. Keep it in that existing metadata block.
4. Validate unknown/missing channels, merged and split output, relevant year
   coverage, blinding, sumw2/systematics, and category-specific removal/skip
   behavior.
5. Update the focused `tests/test_make_cr_and_sr_plots*.py` owner. If a tau
   merged category changes, include
   `test_make_cr_and_sr_plots_1l_tau_merged.py`; if fallback/validation changes,
   use the corresponding channel tests.

An axis or binning change instead belongs to `axes.py` and must keep nominal,
sumw2, and any EFT payloads aligned. A new process group can change displayed
yields and uncertainty bands but not the source PKL; a new physical channel or
processing family usually starts upstream and may require new artifacts.

Histogram processing/fitting edges belong to `topeft/modules/axes.py`, not the
plot metadata. See [change binning](flexible_binning.md). Exact plotting
interfaces and metadata keys are in the
[plotting reference](../reference/plotting.md).

## Submit the maintained plot wrapper to Glados HTCondor

`analysis/topeft_run2/submit_plotter_condor.sh` constructs a Condor description
whose entry script calls `run_plotter.sh` from a shared checkout. It requires a
worker-visible repository, environment, input PKLs, output directory, and log
directory. It does not transfer the repository (`should_transfer_files = NO`)
or produce histograms.

Inspect the derived wrapper and submit description first:

```bash
./submit_plotter_condor.sh --dry-run \
  --ceph-root /cephfs/GROUP/USER/topeft \
  --conda-prefix /cephfs/GROUP/USER/mambaforge/envs/clib-env \
  --request-cpus 2 --request-memory 6GB \
  --log-dir /cephfs/GROUP/USER/topeft/logs \
  -f /cephfs/GROUP/USER/topeft/pickles/plotsCR_Run3.pkl.gz \
  -o /cephfs/GROUP/USER/topeft/plots/run3_combo \
  -y run3 --variable lj0pt --variable ptz --variable ptll
```

Remove `--dry-run` only when the checkout, environment, data, log, and output
paths are visible from the execute node. The helper validates positive CPU and
nonempty memory requests, records its commands, and forwards plotting options
through `run_plotter.sh`. Site credentials, tokens, queue policy, and storage
permissions remain operator responsibilities.

Logs are written as `plotter.<cluster>.<proc>.{log,out,err}` below `--log-dir`.
Use them to distinguish Condor/environment failures from plotter/PKL failures.
The figures appear directly below the forwarded `--output-dir`; there is no
separate retrieval step when shared storage is used.

`submit_plotter_condor.sh` owns Condor resource/path validation and submit-file
construction; `condor_plotter_entry.sh` owns worker-side directory/environment
activation; `run_plotter.sh` and the direct plotter retain plotting semantics.
An extension must preserve those boundaries and the invariants in
`analysis/topeft_run2/test/test_submit_plotter_condor.py`.

## Common failures

| Symptom | Correct owner or action |
| --- | --- |
| wrapper cannot infer CR/SR | pass one explicit region flag; do not rename a PKL as evidence of contents |
| requested channel is absent | inspect artifact channel labels and plotting metadata; do not silently drop a required production category |
| combined channels resolve incompatible fitting axes | correct exact channel binning definitions or plot separately |
| missing/ambiguous sumw2 dense axis | correct the companion artifact/schema; do not use a nominal histogram as variance |
| memory growth with `--workers` | reduce workers; parallel processes each load input state |
| Condor job starts but cannot import/read | fix worker-visible checkout, environment, payload, or storage permissions before resubmission |

To add a physics observable or category, use
[categories and observables](categories_and_observables.md) before updating the
plot metadata. To change only region context, group membership, plotted
variables, a supported binning view, or coverage validation, keep the change in
this guide and the canonical
[`cr_sr_plots_metadata.yml`](../../topeft/params/cr_sr_plots_metadata.yml).

## Output names and diagnostic boundary

Select the processing or fitting view explicitly. Plot outputs and negative
reports carry the corresponding `_processing` or `_fitting` suffix, so the two
views must not share an output namespace. The plotter's local `--workers`
option controls local rendering work; it is unrelated to external Work Queue
worker provisioning.

An `empty-mc-content` skip is a rendering diagnostic for the selected plot
input. It does not establish the content of a bin, a statistical conclusion, or
the adequacy of a fitting view.
