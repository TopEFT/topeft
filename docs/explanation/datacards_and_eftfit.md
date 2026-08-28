# Datacards, scalings, and the EFTFit/Combine boundary

Card production and statistical-workspace construction are separated so each
repository owns one stable boundary. `topeft` converts compatible histogram
artifacts into individual physical-channel cards and templates, selects Wilson
coefficients, and finalizes the channel-aware EFT scaling payload. EFTFit and
Combine later combine those cards and construct the workspace.

## Artifact sequence

`make_cards.py` writes one text card and ROOT template per selected physical
channel, plus `selectedWCs.txt` and `scalings-preselect.json`. The scaling
preselection is still expressed in physical channel names and retains the
producer-owned process, parameter, and coefficient payload.

`datacards_post_processing.py <datacard_dir> -a` selects the current full
topology from `ch_lst.json`. It sorts physical channel names deterministically,
maps them to `ch1`, `ch2`, and so on, copies the selected individual artifacts,
and relabels every matching scaling record while preserving its other fields.
The result is `scalings.json`.

## Repository boundary

`combinedcard.txt` is neither an input nor an output of the finalizer. It is
created later when EFTFit/Combine combines the individual cards. The shared
contract is the deterministic physical-channel order and the compatible set of
individual cards, templates, selected Wilson coefficients, and final scaling
records.

A missing final scaling record means that exact channel/process pair has no
external EFT morph. It is not a process-wide fallback or normalization.

The campaign-specific `run_make_cards_run3_yawen_matrix.sh` is a DATACARD023
archival operator record. It does not own the durable region-to-distribution
mapping or define a supported current wrapper. Changing, generalizing, moving,
or deleting that runnable script requires a separate source-control decision.

See the [card and scaling how-to](../how_to/datacards_and_scalings.md) and the
[artifact reference](../reference/datacards_and_scalings.md).
