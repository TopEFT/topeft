# Data-driven estimation

The maintained data-driven layer turns certified source histograms into
nonprompt or charge-flip products. The processor prepares the source roles and
weights; `run_data_driven.py` and the data-driven modules enforce the product,
schema, variation, and provenance contracts.

## Source and target roles

Each transformation is defined by a requested product, applicable era, named
contributors, and target process. These identities are analysis policy. A
histogram process label does not authorize a transformation unless the
validated artifact and product contract agree.

## Nonprompt transformation

The nonprompt product combines the selected data contribution with configured
prompt-simulation subtraction. Prompt EFT content is evaluated at the SM point
where required by the contract. The transformation preserves the certified
fake-factor variations and propagates the required statistical second moments
rather than inventing a missing variance.

The transformation mechanics and admissibility rules are established in
`topeft/modules/dataDrivenEstimation.py`,
`topeft/modules/data_driven_products.py`, the nominal schema, and the artifact
writer/validator. The repository establishes this implemented product
contract; it does not establish the broader scientific motivation for choosing
the estimator or deriving each rate payload.

## Charge-flip transformation

The charge-flip path selects its configured data source and produces the
maintained target identity. Its product policy determines which nonnominal
content is retained or removed. It is not an implicit secondary producer
hidden behind a process rename; product applicability and sidecar content are
validated explicitly.

## Payloads and variations

Fake-rate, flip-rate, trigger, and related analysis-owned payloads are selected
by era and consumer. Their checked-in files and selectors are the canonical
implementation authorities. Numeric payload content is not duplicated here,
and using a payload does not establish its scientific derivation.

Object and rate variations prepared by the processor must survive or be
removed according to the named product policy. Sumw2 companions follow the
certified statistical contract independently of physics nuisance labels.

## Downstream products

The transformed histogram and sidecar record their source lineage, resolved
product, content families, and validation status. Plotting and card consumers
may use the product only when their required nominal, variation, provenance,
and statistical content is present. See
[histogram artifacts](histogram_artifacts.md) and [sumw2](sumw2.md).

Data-driven production is separate from the main processor execution even when
a wrapper schedules it immediately afterward. That boundary makes a failed
transformation recoverable without relabeling an incomplete source artifact as
a complete product.

## Concrete product anchor and modification route

The source-to-product chain is: processor source roles and fake/flip weights →
validated source histogram/sidecar → `run_data_driven.py` → certified
transformed histogram/sidecar. The canonical policy owners are
[`dataDrivenEstimation.py`](../../topeft/modules/dataDrivenEstimation.py),
[`data_driven_products.py`](../../topeft/modules/data_driven_products.py),
[`nonprompt_policy.py`](../../topeft/modules/nonprompt_policy.py), and
[`run_data_driven.py`](../../analysis/topeft_run2/run_data_driven.py).

There is no single product default independent of source role and region. The
nonprompt path is a representative source-to-transformed product; charge flip
is a separate supported resolved product. Use
[produce or extend nonprompt and charge-flip histograms](../how_to/nonprompt.md)
for payload, role, product-policy, and validation changes.
