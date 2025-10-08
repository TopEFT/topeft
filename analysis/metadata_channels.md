# Metadata Channel and Application Structure

## Source
Information extracted from `topeft/params/metadata.yml` (sections `channels`, `applications`, and `channel_applications`).

## Structure Overview
- **`channels`**
  - YAML sequence of channel identifiers.
  - Ordered list preserves physics category groupings (2lss, 3l, 4l).
  - Entries:
    1. `2lss_m_4j`
    2. `2lss_m_5j`
    3. `2lss_m_6j`
    4. `2lss_m_7j`
    5. `2lss_p_4j`
    6. `2lss_p_5j`
    7. `2lss_p_6j`
    8. `2lss_p_7j`
    9. `3l_onZ_1b`
    10. `3l_onZ_2b`
    11. `3l_m_offZ_1b`
    12. `3l_m_offZ_2b`
    13. `3l_p_offZ_1b`
    14. `3l_p_offZ_2b`
    15. `4l`
- **`applications`**
  - YAML sequence of boolean-flag names describing selection regions.
  - Entries appear in assumed priority order for evaluation: `isAR_2lSS_OS`, `isSR_2lSS`, `isAR_2lSS`, `isSR_3l`, `isAR_3l`, `isSR_4l`.
- **`channel_applications`**
  - Mapping keyed by channel identifiers (same spellings as `channels`).
  - Values are YAML sequences of application flags that are considered valid for each channel.
  - Ordering within each channel list matches `applications` ordering where relevant.

## Region-to-Application Flag Mapping
- **2lSS channels** (`2lss_m_*`, `2lss_p_*`)
  - Associated applications: `isAR_2lSS_OS`, `isSR_2lSS`, `isAR_2lSS` (in that order).
  - Implies that both same-sign signal (`isSR_2lSS`) and two control regions (`isAR_2lSS_OS`, `isAR_2lSS`) are valid per jet multiplicity and charge category.
- **3l channels** (`3l_onZ_*`, `3l_m_offZ_*`, `3l_p_offZ_*`)
  - Associated applications: `isSR_3l`, `isAR_3l`.
  - `onZ` vs `offZ` and charge asymmetry (`m`/`p`) plus b-jet multiplicity determine channel, but application flags do not distinguish these subcategories.
- **4l channel** (`4l`)
  - Associated application: `isSR_4l` only (no control-region applications defined).

## Implicit Assumptions Identified
1. **Channel list mirrors channel keys in mapping.** The `channel_applications` dictionary assumes the `channels` list is the authoritative source for channel naming and ordering; duplication suggests manual synchronization.
2. **Application ordering indicates precedence.** Repeated ordering between `applications` and per-channel lists implies evaluation priority (signal before control or vice versa) though not explicitly encoded.
3. **Shared application flags across related channels.** For every 2lSS channel regardless of charge (`m`/`p`) or jet count (`4j`-`7j`), the same trio of applications is valid, implying uniform region definitions across subchannels.
4. **No year or era dependence.** Regions and their application flags are assumed to be global across all data-taking years.
5. **4l region lacks an associated control region.** Absence of `isAR_4l` suggests either such a control region does not exist or is handled outside this metadata.
6. **3l applications do not differentiate onZ/offZ or charge.** Control/signal flags are reused for all 3l channels, implying that any onZ/offZ or charge-specific treatment occurs elsewhere.

