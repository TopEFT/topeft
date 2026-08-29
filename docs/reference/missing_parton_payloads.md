# Missing-parton correction payloads

## Installed files and selection

The datacard consumer selects:

- `topeft/data/missing_parton/missing_parton_run2.root` for `UL16`,
  `UL16APV`, `UL17`, and `UL18`;
- `topeft/data/missing_parton/missing_parton_run3.root` for `2022`, `2022EE`,
  `2023`, and `2023BPix`.

`make_cards.py --miss-parton-file PATH` supplies an exact replacement. The
consumer does not search for another layout or silently fall back to a
different era. `--skip-missing-parton-rate-syst` disables only this nuisance;
disabling all nuisances also avoids opening the payload.

The consumer methods in `DatacardMaker` are developer-facing:

| Symbol | Parameters/defaults and return | Contract |
| --- | --- | --- |
| `DatacardMaker.missing_parton_run_era` | One canonical year/period → `run2` or `run3` | Rejects missing and unsupported labels. |
| `DatacardMaker.missing_parton_run_era_for_years` | String or iterable; optional payload path for diagnostics → one era | Rejects empty or mixed Run 2/Run 3 selections. |
| `DatacardMaker.missing_parton_nuisance_name_for_years` | Same year inputs → `missing_parton` | Validates era before returning the single correlated nuisance name. |
| `DatacardMaker.resolve_missing_parton_payload_path` | Years, optional exact path, registry default current → path | Explicit non-empty path wins. Implicit defaults are allowed only for the current registry; a custom registry requires a matching explicit payload. |
| `DatacardMaker.load_systematics` | Rate JSON path and resolved missing-parton path → systematic mapping | Reads payload only when nuisances are enabled and missing-parton is not skipped; parsing/schema/lookup errors fail card construction. |

These consumers read but never modify the installed ROOT files.

## Current schema

Each payload has exactly one top-level `TTree` per `ALL_CH_LST_SR` base
category, in registry order. Each tree contains one `tllq`
`double`/`float64` array branch. Array indices are physical jet
multiplicities. For a terminal registry category `>N`, index `N` represents the
complete `njet >= N` population and no index above `N` is stored. Values must
be finite.

Current semantic digests, using the serialization defined by
`tests/test_missing_parton_payload_schema.py`, are:

- Run 2: `936a7316894257a5dcac31c345c60ea273d27cb672c71fbce6382fe5df534a24`
- Run 3: `8ddf59420ed47828551803ef7b168ae1dec02e1402418801ab5ec2efc90de332`

The only supported legacy layout is the schema introduced by immutable
`run3_test_mmerged` commit
`2469053a8d7ab0b42c86c68000f51b6e7f6dafff`. It is a bounded read/test
compatibility contract, not a current production output format.

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
