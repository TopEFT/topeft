# Missing-parton correction payloads

## Installed files and selection

The datacard consumer selects:

- `topeft/data/missing_parton/missing_parton_run2.root` for `UL16`,
  `UL16APV`, `UL17`, and `UL18`;
- `topeft/data/missing_parton/missing_parton_run3.root` for `2022`, `2022EE`,
  `2023`, and `2023BPix`.

`make_cards.py --miss-parton-file PATH` supplies an exact replacement. The
consumer infers its layout from the payload's ordered top-level trees and array
lengths against the five maintained layouts. No match or multiple matches
fail; there is no fallback. The card CLI has no `--sr-registry` option.
`--skip-missing-parton-rate-syst` disables only this nuisance;
disabling all nuisances also avoids opening the payload.

The consumer methods in `DatacardMaker` are developer-facing:

| Symbol | Parameters/defaults and return | Contract |
| --- | --- | --- |
| `DatacardMaker.missing_parton_run_era` | One canonical year/period → `run2` or `run3` | Rejects missing and unsupported labels. |
| `DatacardMaker.missing_parton_run_era_for_years` | String or iterable; optional payload path for diagnostics → one era | Rejects empty or mixed Run 2/Run 3 selections. |
| `DatacardMaker.missing_parton_nuisance_name_for_years` | Same year inputs → `missing_parton` | Validates era before returning the single correlated nuisance name. |
| `DatacardMaker.resolve_missing_parton_payload_path` | Years, optional exact path → path | Explicit non-empty path wins; otherwise the run-era default is selected. Empty paths and mixed eras fail. |
| `DatacardMaker.load_systematics` | Rate JSON path and resolved missing-parton path → systematic mapping | Reads payload only when nuisances are enabled and missing-parton is not skipped; parsing/schema/lookup errors fail card construction. |
| `missing_parton_contract.infer_legacy_missing_parton_layout` | Payload path → one maintained layout | Matches ordered tree names and stored lengths; no match and ambiguity fail. |

These consumers read but never modify the installed ROOT files.

## Current schema

The installed default payloads use `ALL_CH_LST_SR`: one top-level `TTree` per
base category, in registry order. Other maintained layouts are accepted when
their stored tree order and lengths match exactly. Each tree contains one `tllq`
`double`/`float64` array branch. Array indices are physical jet
multiplicities. For a terminal registry category `>N`, index `N` represents the
complete `njet >= N` population and no index above `N` is stored. Values must
be finite.

Current semantic digests, using the serialization defined by
`tests/test_missing_parton_payload_schema.py`, are:

- Run 2: `936a7316894257a5dcac31c345c60ea273d27cb672c71fbce6382fe5df534a24`
- Run 3: `8ddf59420ed47828551803ef7b168ae1dec02e1402418801ab5ec2efc90de332`

The payload producer is a separate workflow. Its
`--missing-parton-layout-key` option selects the output layout during
generation; the datacard consumer infers the layout when reading an existing
file.

## Source provenance and maintenance boundary

The Run 2 file derives from the accepted Run 2 all-analysis missing-parton
source-card production. The Run 3 file derives from the accepted Run 3
fixyield production used by the maintained workflow. The accepted Run 2
source-card and corresponding maintained production used the same upstream
histogram PKLs; this does not assert byte identity for unavailable private
inputs.

Source manifests must identify each consumed ROOT/TXT pair, category and role,
and content hash. Reproduction paths that are specific to a storage site are
environment details, not fields in the installed payload schema.

Replacing a payload requires deterministic semantic validation of tree set and
order, array lengths, finite values, physical jet indexing, terminal-bin
construction, and exact consumer lookup. It is a production/payload action,
not a documentation operation.

## Source and test authority

- `topeft/modules/missing_parton_contract.py`
- `topeft/modules/datacard_tools.py`
- `analysis/topeft_run2/make_cards.py`
- `tests/test_missing_parton_contract.py`
- `tests/test_missing_parton_payload_schema.py`
- `tests/test_missing_parton_payload_roundtrip.py`
- `tests/test_missing_parton_sr_registry.py`

For the physics model, see the
[missing-parton uncertainty explanation](../explanation/missing_parton_uncertainties.md).
For replacement and validation steps, use the
[missing-parton payload how-to](../how_to/missing_parton_payloads.md). The
[datacard and scaling reference](datacards_and_scalings.md) describes the
consumer boundary in the wider card workflow.
