# Metadata Channel and Application Structure

## Source
Information extracted from the grouped schema embedded in `topeft/params/metadata.yml` under `channels.groups` and `channels.scenarios`. The content mirrors `analysis/channels_schema.yaml` so that metadata consumers no longer need to cross-reference separate lists of channels and applications.

## Group Overview
Each entry in `channels.groups` represents a coherent collection of regions with shared descriptions, jet-bin definitions, and application tags. Scenario lists attached to each group indicate where the configuration is valid.

- **`TOP22_006_CH_LST_SR`** – Baseline Run 2 ttH multilepton signal regions for the TOP-22-006 reinterpretation. Provides 2ℓSS (charge-split, 4–≥7 jets), 3ℓ (on-Z/off-Z split with 1–2 b-tags), and 4ℓ categories with both signal and associated control application tags for MC and data.【F:topeft/params/metadata.yml†L9-L74】
- **`TAU_CH_LST_SR`** – Tau-enhanced signal regions extending the baseline bins with explicit tau-enriched selections (0τ, 1τ, and opposite-sign tau channels) plus corresponding jet bins and application tags.【F:topeft/params/metadata.yml†L75-L178】
- **`TAU_CH_LST_CR`** – Control regions supporting the tau analysis, including fake-tau and real-tau validation bins alongside 1ℓ1τ control categories.【F:topeft/params/metadata.yml†L179-L226】
- **`OFFZ_SPLIT_CH_LST_SR`** – Alternative signal-region grouping with finer 3ℓ off-Z splitting (low/high/none) at both 1b and 2b levels, while keeping the standard 2ℓSS and 4ℓ bins.【F:topeft/params/metadata.yml†L227-L362】
- **`CH_LST_CR`** – Shared control regions for fake, charge-flip, and validation studies, covering 2ℓSS/OS validation, Zee veto variants, and inclusive 3ℓ control bins.【F:topeft/params/metadata.yml†L363-L498】
- **`FWD_CH_LST_SR`** – Forward-enriched signal regions extending the 2ℓSS categories with forward-jet tags while reusing the multilepton jet binning and application tags.【F:topeft/params/metadata.yml†L499-L596】

## Scenario Overview
`channels.scenarios` documents how channel groups combine for different analyses:

- **`TOP_22_006`** – Combines baseline signal regions, shared control regions, and the off-Z trilepton split.【F:topeft/params/metadata.yml†L597-L606】
- **`tau_analysis`** – Pairs tau-enriched signal and control regions with the shared control set.【F:topeft/params/metadata.yml†L607-L613】
- **`fwd_analysis`** – Uses forward-focused signal regions plus the common control suite.【F:topeft/params/metadata.yml†L614-L619】

Consumers can use these scenario definitions to pick the appropriate set of channel groups without manually synchronising individual channel lists and application flags.

