# Metadata Channel and Application Structure

## Source
Information extracted from the grouped schema embedded in `topeft/params/metadata.yml` under `channels.groups` and `channels.scenarios`. The content mirrors `analysis/channels_schema.yaml` so that metadata consumers no longer need to cross-reference separate lists of channels and applications.

## Group Overview
Each entry in `channels.groups` represents a coherent collection of regions with shared descriptions, jet-bin definitions, and application tags. Scenario lists attached to each group indicate where the configuration is valid. Optional `features` lists document metadata flags (e.g., off-Z splits or tau requirements) that downstream tools can toggle on.

- **`TOP22_006_CH_LST_SR`** – Baseline Run 2 ttH multilepton signal regions for the TOP-22-006 reinterpretation. Provides 2ℓSS (charge-split, 4–≥7 jets), 3ℓ (on-Z/off-Z split with 1–2 b-tags), and 4ℓ categories with both signal and associated control application tags for MC and data.【F:topeft/params/metadata.yml†L10-L120】
- **`TAU_CH_LST_SR`** – Tau-enhanced signal regions extending the baseline bins with explicit tau-enriched selections (0τ, 1τ, and opposite-sign tau channels) plus corresponding jet bins and application tags. Declares the `requires_tau` feature flag for easy discovery.【F:topeft/params/metadata.yml†L121-L312】
- **`TAU_CH_LST_CR`** – Control regions supporting the tau analysis, including fake-tau and real-tau validation bins alongside 1ℓ1τ control categories.【F:topeft/params/metadata.yml†L312-L358】
- **`OFFZ_SPLIT_CH_LST_SR`** – Alternative signal-region grouping with finer 3ℓ off-Z splitting (low/high/none) at both 1b and 2b levels, while keeping the standard 2ℓSS and 4ℓ bins. Tagged with the `offz_split` feature flag.【F:topeft/params/metadata.yml†L359-L448】
- **`CH_LST_CR`** – Shared control regions for fake, charge-flip, and validation studies, covering 2ℓSS/OS validation, Zee veto variants, and inclusive 3ℓ control bins.【F:topeft/params/metadata.yml†L453-L549】
- **`FWD_CH_LST_SR`** – Forward-enriched signal regions extending the 2ℓSS categories with forward-jet tags while reusing the multilepton jet binning and application tags. Advertises the `requires_forward` feature flag.【F:topeft/params/metadata.yml†L551-L673】

## Scenario Overview
`channels.scenarios` documents how channel groups combine for different analyses:

- **`TOP_22_006`** – Combines baseline signal regions, shared control regions, and the off-Z trilepton split.【F:topeft/params/metadata.yml†L675-L683】
- **`tau_analysis`** – Pairs tau-enriched signal and control regions with the shared control set.【F:topeft/params/metadata.yml†L684-L690】
- **`fwd_analysis`** – Uses forward-focused signal regions plus the common control suite.【F:topeft/params/metadata.yml†L691-L696】

Consumers can use these scenario definitions to pick the appropriate set of channel groups without manually synchronising individual channel lists and application flags.

### Selecting scenarios in `run_analysis.py`
The command-line front-end now accepts ``--scenario`` to activate one or more of the
metadata-defined combinations above (defaulting to ``TOP_22_006`` when unspecified).
Additional channel groups can be pulled in by advertising feature tags via
``--channel-feature`` (for example ``--channel-feature requires_tau``). The helper
functions resolve the requested scenarios and features to determine which regions and
systematics should run.

## Histogram variables
Histogram requests now mirror every entry declared under ``metadata['variables']`` so
the runner no longer depends on metadata-side profile aliases. Per-channel include and
exclude lists remain available through ``histogram_variables`` blocks and are solely
responsible for scoping observables to specific regions.【F:topeft/params/metadata.yml†L1270-L1290】

