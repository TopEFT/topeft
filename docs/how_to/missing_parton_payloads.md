# Regenerate and validate missing-parton payloads

This guide covers the maintained operator workflow for deriving a
missing-parton correction payload from accepted central `tZq` and private
`tllq` source cards. Payload generation is not event processing: it begins with
matched per-category ROOT/TXT card pairs and ends with a compact ROOT correction
consumed later by `DatacardMaker`.

The source authority is:

- producer CLI and derivation: `analysis/topeft_run2/missing_parton.py`;
- registry and schema contract: `topeft/modules/missing_parton_contract.py` and
  `topeft/channels/ch_lst.json`;
- consumer selection and lookup: `topeft/modules/datacard_tools.py` and
  `analysis/topeft_run2/make_cards.py`;
- installed files: `topeft/data/missing_parton/missing_parton_run2.root` and
  `topeft/data/missing_parton/missing_parton_run3.root`.

Read the [physics and architecture explanation](../explanation/missing_parton_uncertainties.md)
for the residual-uncertainty definition, directional quadrature, terminal-bin
semantics, and process/correlation policy. Use the
[payload reference](../reference/missing_parton_payloads.md) for exact installed
file selection, schema, semantic digests, and validation owners.

## 1. Freeze accepted source inputs

Choose one accepted central-`tZq` source-card directory and one accepted
private-`tllq` source-card directory for a single era. Each directory must
contain the matched files
`ttx_multileptons-<base_category>_njets.root` and `.txt` for every base category
in `ALL_CH_LST_SR`.

Before generation, create a production manifest recording for every pair:

- era and base category;
- central or private role and expected process identity;
- absolute or campaign-relative source location;
- content hash of both ROOT and TXT inputs; and
- the registry/source commit and generation environment.

Storage paths are environment details, not payload-schema fields. The hashes,
roles, category identity, and registry establish the reproducible source set.
Stop if an input hash changes without an accepted source-production decision.

## 2. Validate the complete derivation without writing

Run from the repository root in the configured analysis environment. Use an
explicit fresh scratch destination even for a dry run:

```bash
central_card_dir=/path/to/accepted/central_tzq
private_card_dir=/path/to/accepted/private_tllq
scratch_payload=/path/to/writable/scratch/missing_parton_run2.root

python analysis/topeft_run2/missing_parton.py \
  --central-card-dir "${central_card_dir}" \
  --private-card-dir "${private_card_dir}" \
  --sr-registry ALL_CH_LST_SR \
  --output-file "${scratch_payload}" \
  --var njets \
  --dry-run
```

`--central-card-dir` and `--private-card-dir` must be supplied together.
Payload production supports only `--var njets`. The dry run validates the full
category inventory, ROOT/TXT pairing, process identities, physical eight-bin
`njets` axes, numerical inputs, registry layout, terminal categories, and
planned payload without writing the ROOT file.

A successful dry run authorizes neither a write nor installation. Resolve every
reported missing/extra category, schema mismatch, non-finite value, process
mismatch, or source-manifest drift at its owning layer before continuing.

## 3. Write a fresh scratch payload

Repeat the same command without `--dry-run`:

```bash
python analysis/topeft_run2/missing_parton.py \
  --central-card-dir "${central_card_dir}" \
  --private-card-dir "${private_card_dir}" \
  --sr-registry ALL_CH_LST_SR \
  --output-file "${scratch_payload}" \
  --var njets
```

The producer refuses an existing output by default. Prefer a new scratch path.
Use `--overwrite` only when replacement of that exact scratch artifact is
authorized; the writer validates a temporary file before atomically replacing
the target. Do not point exploratory generation at either installed payload.

The write step owns construction of a candidate payload only. It does not
install the file, change consumer defaults, or establish that Run 2 and Run 3
remain a coherent pair.

## 4. Validate semantics independently

Before considering installation:

1. Recompute and compare every source-card hash with the frozen manifest.
2. Require the exact `ALL_CH_LST_SR` tree set in registry order.
3. Require one `tllq` `double`/`float64` array branch per tree with the current
   registry-derived lengths.
4. Require finite, non-negative values, physical-jet indexing, and the current
   inclusive terminal layout. Current forward arrays have length five; the
   bounded historical compatibility layout has length six and is not a current
   producer target.
5. Generate twice from the same frozen inputs and compare semantic identities,
   not raw ROOT container bytes.
6. Independently reconstruct every direct and inclusive-terminal value without
   calling the producer's payload-plan or writer implementation.
7. Exercise exact consumer lookup and era selection.

Run the focused schema, registry, contract, option, and round-trip tests:

```bash
python -m pytest -q \
  tests/test_missing_parton_payload_schema.py \
  tests/test_missing_parton_registry_layout.py \
  tests/test_missing_parton_contract.py \
  tests/test_make_cards_missing_parton_option.py \
  tests/test_missing_parton_payload_roundtrip.py
```

Those tests check the repository contracts. They do not replace source-manifest
verification, deterministic regeneration, or independent numerical
reconstruction for a newly produced candidate.

## 5. Treat installation as a separate payload action

Installation replaces tracked binary correction inputs used by card
production. It therefore requires explicit payload-generation and installation
authorization beyond documentation work.

When authorized, treat the Run 2 and Run 3 candidates as one installation unit:

1. complete the freeze, dry-run, scratch-write, deterministic regeneration,
   independent reconstruction, strict schema, and consumer checks for both;
2. compare each candidate's semantic identity with the expected production
   record and explain every change;
3. replace exactly `missing_parton_run2.root` and
   `missing_parton_run3.root`, without changing the public filenames or adding
   fallback discovery;
4. reopen the installed files and repeat schema, semantic-identity,
   reconstruction, era-selection, and exact-lookup checks; and
5. commit the payloads with their source manifests, hashes, validation evidence,
   and declared consumer impact under the separately authorized workflow.

Do not install one era merely because the other was unchanged, infer the schema
from the candidate under test, add automatic current/legacy detection, or
partially migrate the producer and consumer.

## 6. Select a candidate payload without installing it

For a bounded card check, pass an exact package-relative payload path:

```bash
python analysis/topeft_run2/make_cards.py input_histograms.pkl.gz \
  --year 2022 --do-nuisance \
  --miss-parton-file data/missing_parton/candidate_run3.root \
  --var-lst ht --ch-lst 3l_onZ_1b_2j
```

The explicit path is resolved relative to the installed `topeft` package. The
consumer does not inspect an invalid candidate and fall back to the era default.
For an external scratch file, place or expose it only through the separately
authorized validation setup rather than changing default selection logic.

To retain other nuisances while intentionally omitting this one, use
`--skip-missing-parton-rate-syst`. Omitting `--do-nuisance` also prevents
rate-systematic payload loading. Neither option is a substitute for validating
a candidate that will be installed.

## 7. Diagnose failures at the owning layer

| Symptom | Inspect first |
| --- | --- |
| missing or extra tree/category | frozen ROOT/TXT inventory and `ALL_CH_LST_SR` selection |
| tree-order mismatch | registry-order writer; do not alphabetically sort the public contract |
| wrong array length or forward terminal | registry `=N`/`>N` metadata and current terminal aggregation |
| non-finite or negative fraction | nominal, shape, and rate inputs before the residual formula |
| wrong process application | consumer mapping, which must apply only to `tllq` and `tHq` |
| payload opened when disabled | `--do-nuisance` and `--skip-missing-parton-rate-syst` before path resolution |
| semantic digest mismatch | source hashes, registry order, lengths, derivation, and every stored value |

Do not repair these failures by editing the ROOT payload manually. Correct the
source inputs, registry, producer, or consumer at its authority and regenerate
the candidate through the validated path.
