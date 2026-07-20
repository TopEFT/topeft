# Missing-parton correction payloads

This directory contains the correction payloads used to write the
`missing_parton` normalization nuisance in analysis datacards:

- `missing_parton_run2.root` for `UL16`, `UL16APV`, `UL17`, and `UL18`;
- `missing_parton_run3.root` for `2022`, `2022EE`, `2023`, and `2023BPix`.

For the physics definition, derivation formula, terminal-bin construction,
consumer behavior, reproduction procedure, and troubleshooting guidance, see
the [missing-parton uncertainties guide](../../../docs/missing_parton_uncertainties.md).

## Source authority

- Run 2 uses the accepted Run 2 all-analysis missing-parton source-card
  production.
- Run 3 uses source cards from the accepted Run 3 fixyield production used by
  the maintained analysis workflow.
- The accepted Run 2 source-card production and the corresponding maintained
  Run 2 production used the same upstream histogram PKLs. This provenance
  statement does not assert byte identity for unavailable private PKL inputs.

Source manifests must identify every consumed ROOT/TXT pair, its category and
role, and a content hash. Site-specific storage paths are environment details,
not part of the public payload contract.

## Current public schema

Each payload contains exactly one top-level `TTree` for every
`ALL_CH_LST_SR` base category, in registry order. Each tree has one `tllq`
`double`/`float64` array branch. Array indices are physical jet multiplicities.
For a terminal registry category `>N`, index `N` represents the complete
`njet >= N` population and no index above `N` is stored. All values must be
finite.

The current semantic digests are:

- Run 2: `4a869bc8ecc56adb491100e50b29d0e600a6916824b2849ff9f9d31c5a09736a`
- Run 3: `6f948c7859a43249dae70e4e679c4439425dc4384991ab2829e0d85c09eed26f`

`tests/test_missing_parton_payload_schema.py` defines the serialization used
for these semantic identities and checks both installed files.

## Supported legacy schema

The only supported legacy compatibility contract is the schema introduced by
immutable `run3_test_mmerged` commit
`2469053a8d7ab0b42c86c68000f51b6e7f6dafff`. It is covered by a separate,
explicit test schema. No other historical or partially migrated layout is
supported, and the legacy test contract is not a production output format.

## Consumer selection

With nuisances enabled, `DatacardMaker` selects the Run 2 or Run 3 file above
from the requested years. `--miss-parton-file` supplies an exact replacement
path; the consumer does not search for another layout or fall back to an era
default. `--skip-missing-parton-rate-syst` disables only this nuisance and does
not open a missing-parton payload. Disabling all nuisances also avoids opening
the payload.

## Maintenance

Replace these files only from accepted, hashed source manifests. Regeneration
must be deterministic at the semantic level and must validate tree set, tree
order, array lengths, finite values, physical-jet indexing, and exact consumer
lookup. Reconstruct direct and terminal values independently, run focused
consumer checks, and replace both era files together when both are ready.
