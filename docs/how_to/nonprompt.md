# Produce nonprompt and charge-flip histograms

`run_data_driven.py` transforms a completed processor histogram artifact into a
new artifact containing the applicable nonprompt and charge-flip products. It
does not rerun NanoAOD processing, and the source PKL is not interchangeable
with the transformed `_np.pkl.gz` output expected by downstream cards.

The direct entry point owns input-artifact validation, data-driven application,
streaming/materialized serialization choice, derived companion requirements,
lineage, and transformed sidecar publication. It delegates histogram-level
estimation to `DataDrivenProducer`. It does not own event selection, the source
sumw2 policy, application-axis definitions, or card/plot selection.

| Setting | Source/default | Effect |
| --- | --- | --- |
| input | required `--input-pkl` | must be a readable processor output; its canonical adjacent sidecar is discovered automatically |
| output | derived when `--output-pkl` is omitted | removes `.pkl[.gz]` and appends `_np.pkl.gz` (or `_np_nominal_reference.pkl.gz`) |
| memory path | streaming by default | hard-coded pickle protocol 3 and memo-clear interval 1; `--legacy-dict-mode` materializes the dictionary |
| progress | 30-second heartbeat | heartbeat, quiet, and memory-report options change diagnostics, not histogram semantics |
| product restriction | full applicable family by default | `--only-flips` removes nonprompt output; nominal-only reference is explicitly non-card-ready |

## Choose inline or deferred execution

For a direct `run_analysis.py` invocation:

```bash
python run_analysis.py INPUT.cfg --do-np --np-postprocess=inline \
  --outpath /absolute/path/to/output --outname analysis_block
```

`inline` is the default when `--do-np` is active. It publishes the source and
transformed products in one invocation. For campaign production, use the
deferred lifecycle so the transform runs in a fresh process and can be resumed
independently:

```bash
python run_analysis.py INPUT.cfg --do-np --np-postprocess=defer \
  --outpath /absolute/path/to/output --outname analysis_block

python run_data_driven.py \
  --input-pkl /absolute/path/to/output/analysis_block.pkl.gz \
  --output-pkl /absolute/path/to/output/analysis_block_np.pkl.gz
```

`--np-postprocess=skip` creates no transformed artifact. Passing
`--defer-np` to `fullR3_run.sh` without `--do-np` likewise does not enable the
producer.

Do not pass a sidecar path manually. The helper rejects the removed
`--metadata-json` interface and derives the canonical sidecar from the input
PKL. This prevents a metadata record from one artifact being paired with a
different histogram dictionary.

## Recover after a post-processing failure

If event processing completed and the source PKL is valid, choose a fresh
output path and rerun only `run_data_driven.py`. Do not rerun the processor or
Work Queue merely to repeat the transformation. The helper uses its streaming
iterator path by default; `--legacy-dict-mode` requests the older
materialized-dictionary path.

Before retrying, verify that the source artifact and its metadata sidecar still
refer to one another, that the desired output does not contain a partial prior
result, and that the required sumw2 companions are present. Sidecar
publication is part of successful transformation; a PKL without its expected
provenance is not a complete maintained output.

The source must validate as `processor_output`. Current versioned artifacts
carry a resolved data-driven contract, sumw2 provenance, content manifest,
sample contract, and exact PKL identity. The helper derives which source
process/family companions each requested transformation needs and refuses a
missing companion; it does not reconstruct a variance from nominal yields or
weaken requirements in recovery mode. A legacy uniform artifact follows its
explicit bounded legacy path and is not silently upgraded with a current
sidecar.

For long inputs, retain streaming mode. The `protocol=3` and
`clear_memo_interval=1` writer settings are currently hard-coded for the
memo-clearing bounded-memory path, not user-facing tuning knobs. Use
`--legacy-dict-mode` only to compare or recover the historical materialized
implementation, accepting its higher peak memory.

## Understand applicability

Products are selected from the source histogram's application-axis contract:

- nonprompt consumes `isAR_1l`, `isAR_2lSS`, `isAR_2lOS`, and `isAR_3l`;
- flips consumes `isAR_2lSS_OS`.

A family without `isAR_2lSS_OS` does not produce flips. If that label is
present and flips are enabled, the flips output is required. An unknown
`isAR_*` label is not automatically a supported nonprompt region. Change the
application registry, producer behavior, sidecar schema, and tests together;
never edit a sidecar to make an unknown label look supported.

## Extend a supported data-driven product

An extension crosses several maintained contracts:

1. Define the product and exact application regions at
   `topeft/modules/data_driven_products.py`; do not infer support from an
   arbitrary `isAR_*` string.
2. Implement histogram transformation at `DataDrivenProducer`, preserving
   nominal/EFT treatment, prompt-subtraction groups, year/process identity, and
   systematic pairs.
3. Derive required source companions and generated output companions from the
   transformation contract. Do not duplicate a hand-maintained family list in
   `run_data_driven.py`.
4. Extend artifact kind, lineage, resolved contract, content manifest, and
   validation only when the current schema cannot express the product.
5. Preserve streaming and materialized output equivalence.
6. Update `tests/test_data_driven_products.py`,
   `tests/test_run_data_driven.py`,
   `tests/test_run_data_driven_iterator_mode.py`,
   `tests/test_data_driven_streaming.py`,
   `tests/test_histogram_artifact_sidecars.py`, and the affected policy/consumer
   tests.

Validate applicability-present and applicability-absent families, missing
required companions, scalar and EFT source processes, multi-year labels,
sidecar/lineage readback, streaming/materialized equivalence, and downstream
card readiness. The output can affect plotting, card yields, statistical
uncertainties, nuisance shapes, and EFT prompt subtraction.

## Diagnose failures

| Failure | Owning check |
| --- | --- |
| input is not `processor_output` or sidecar identity disagrees | artifact/lineage validation; use the matching source PKL and sidecar |
| requested product is absent from the resolved contract | application-axis/product configuration; do not force a sidecar edit |
| required `*_sumw2` process/family is absent | source sumw2 policy and content manifest; reproduce the source artifact when necessary |
| output exists or a prior write is partial | choose a fresh destination and repeat only after validating the source |
| streaming and legacy paths differ | serialization/transformation regression; retain both results and stop downstream use |
| only-flips or nominal-reference output used for cards | wrong artifact kind; produce the full card-ready transformed product |

Read [sumw2 operations](sumw2.md) before changing companion selection. The
[artifact and provenance explanation](../explanation/artifacts_and_provenance.md)
describes why the transform publishes a new identity, while the
[software reference](../reference/entrypoints.md) records exact CLI contracts.

For a fake/flip payload or evaluator change, use
[corrections, weights, and systematics](corrections_weights_and_systematics.md)
and return here to validate product propagation. For a prompt, conversion,
subtraction, or target-role change, use
[sample roles and normalization](sample_roles_and_normalization.md). The
[data-driven reference](../reference/data_driven_estimation.md) is the direct
semantic/default bridge for low-level product helpers.
