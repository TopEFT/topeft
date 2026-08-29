# Histogram artifacts and provenance

The analysis passes a family of related objects between stages, not an
unqualified pickle file. A source histogram PKL contains the processor's
nominal, EFT, systematic, and policy-selected sumw2 histograms. Its adjacent
metadata sidecar records the evidence needed to decide whether another stage
may consume or combine that content.

## Why the sidecar is separate

The PKL is optimized for histogram operations. The JSON sidecar is optimized
for identity, compatibility, and fail-closed readback. Keeping these jobs
separate lets a consumer inspect production identity, histogram-family scope,
Wilson-coefficient order, sumw2 policy, and transformation lineage before it
loads or combines a large payload.

A filename, directory name, matching number of bins, or nearby campaign log is
not compatibility evidence. Consumers validate the sidecar against the actual
histogram content and their own required families.

## Source, split, and transformed products

`run_analysis.py` and the processor layer create source artifacts. A source
run may publish coherent split-family fragments when the sidecar proves that
they share the same source identity and cover disjoint intended content.
`run_data_driven.py` creates a distinct transformed nonprompt artifact with
lineage back to its source inputs. The `_np.pkl.gz` output is therefore not a
renamed source PKL.

Transformations must preserve the metadata that downstream consumers need:
nominal family relationships, sumw2 companions, EFT coefficients, scaling
semantics, and physical axes. A transformation that preserves only nominal
bin values is incomplete.

## Responsibility boundary

- The processor owns event selection, weights, histogram filling, and the
  initial source contract.
- Artifact helpers own atomic PKL/sidecar publication, schema validation,
  lineage, and merge compatibility.
- `run_data_driven.py` owns transformed nonprompt content and its transformation
  record.
- Plotting and card consumers own their required-family and compatibility
  checks; a producer cannot certify a consumer requirement on their behalf.

See the [artifact reference](../reference/histogram_artifacts.md) for exact
schemas and APIs, the [histogram tutorial](../tutorials/histogram_artifacts.md)
for guided inspection, and the [nonprompt how-to](../how_to/nonprompt.md) for
operations.
