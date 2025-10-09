# Analysis Feature Dependencies

This note summarizes how the Run 2 analysis processor reacts to the channel feature
tags declared in the metadata.  Scenarios selected via ``run_analysis.py``'s
``--scenario`` option or direct ``--channel-feature`` requests enable the behaviors
described below. The information is distilled from `analysis_processor.py` so future
metadata-driven refactors can reproduce the same logic.

## `offz_split`

When enabled, the three-lepton off-Z control region is divided into multiple categories
and several helpers switch behavior:

- **Masks:**
  - Always build the inclusive on-/off-Z masks (`sfosz_3l_OnZ_mask`/`sfosz_3l_OffZ_mask`).
  - With the split enabled also build:
    - `sfosz_3l_OffZ_low_mask` via `tc_es.get_off_Z_mask_low`.
    - `sfosz_3l_OffZ_any_mask` via `tc_es.get_any_sfos_pair`.
  - Register packed-selection entries `3l_offZ_low`, `3l_offZ_high`, and `3l_offZ_none`.
    Otherwise register the single `3l_offZ` mask.
- **Variables:**
  - The nominal `ptz` observable switches from `te_es.get_Z_pt` to `te_es.get_ll_pt` so the
    off-Z sub-categories use the dilepton kinematics.
- **Histogram filling rules:**
  - While looping over histogram fills skip `ptz` templates for categories that are not the new
    off-Z split channels to avoid mismatched definitions.

## `requires_tau`

This feature activates the hadronic-tau analysis branch and propagates through selection,
weights, and observables.

- **Object preparation:**
  - Apply TES/FES corrections (`ApplyTES`, `ApplyTESSystematic`, `ApplyFESSystematic`) and
    decay-mode filtering before cleaning.
  - Cache `cleaning_taus`, tau multiplicities, and the padded leading tau (`tau0`) for later use.
- **Masks registered in `PackedSelection`:**
  - Add preselection masks `1l`, `1tau`, `1Ftau`, `0tau`, `onZ_tau`, and `offZ_tau` to support the
    tau control and signal regions.
- **Jet cleaning:**
  - Include `cleaning_taus` in the object veto collection while building `cleanedJets`.
- **Weights:**
  - Attach tau scale factors to the event record and register `lepSF_taus_real`/`lepSF_taus_fake`
    weights for the 1l, 2l, and 3l channel prefixes.
- **Variables:**
  - Permit tau-sensitive observables by building `l_j_collection` with the tau candidates and, when
    requested, computing `ptz_wtau` for tau-inclusive signal regions.
- **Histogram filling rules:**
  - Restrict `ptz`/`ptz_wtau` templates to the tau-enabled categories so the shapes stay consistent
    with the modified definitions.

## `requires_forward`

This feature exposes forward-jet enriched categories for the same-sign dilepton channels.

- **Masks:**
  - Register `fwdjet_mask` and its complement, then add preselection entries `2lss_fwd`,
    `2l_fwd_p`, and `2l_fwd_m` that require a forward jet along with the standard charge
    selections.
- **Histogram filling rules:**
  - While filling templates skip `ptz` outside on-Z channels (matching the forward configuration)
    and restrict `lt` histograms to the same-sign leptonic categories that include the forward
    requirement.

These notes capture the behaviors that must be replicated when the analysis configuration
migrates to metadata-driven definitions.
