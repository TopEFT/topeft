"""Diagnostics for photons recovered from selected conversion-like leptons.

Recovered means that a gen photon was found. Classified origin means that the
recovered photon has a clean decay or production ancestry; hadron and ambiguous
guard categories are intentionally excluded from classified origin.
"""

import re

import awkward as ak
import numpy as np


NOT_CONVERSION = 0
DECAY_LEPTON = 1
DECAY_W_OR_B_WITH_TOP_ANCESTOR = 2
DECAY_TOP_COPY_CONDITION = 3
PRODUCTION_ISR = 4
PRODUCTION_OFFSHELL_TOP = 5
HADRON_ANCESTOR = 6
AMBIGUOUS = 7
NO_PHOTON_FOUND = 8
INVALID_MATCH = 9

CATEGORY_NAMES = {
    NOT_CONVERSION: "not_conversion",
    DECAY_LEPTON: "decay_lepton",
    DECAY_W_OR_B_WITH_TOP_ANCESTOR: "decay_w_or_b_with_top_ancestor",
    DECAY_TOP_COPY_CONDITION: "decay_top_copy_condition",
    PRODUCTION_ISR: "production_isr",
    PRODUCTION_OFFSHELL_TOP: "production_offshell_top",
    HADRON_ANCESTOR: "hadron_ancestor",
    AMBIGUOUS: "ambiguous_no_mother_or_malformed_chain",
    NO_PHOTON_FOUND: "no_photon_found",
    INVALID_MATCH: "invalid_match",
}

DECAY_CATEGORIES = (
    DECAY_LEPTON,
    DECAY_W_OR_B_WITH_TOP_ANCESTOR,
    DECAY_TOP_COPY_CONDITION,
)
PRODUCTION_CATEGORIES = (PRODUCTION_ISR, PRODUCTION_OFFSHELL_TOP)

EVENT_DIAGNOSTIC_PREFIX = "ttgamma_photon_history_"
DEFAULT_MAX_ANCESTRY_DEPTH = 64

TTGAMMA_PRODUCTION_SAMPLE = "ttgamma_production"
TTGAMMA_DECAY_SAMPLE = "ttgamma_decay"
TTGAMMA_INCLUSIVE_SAMPLE = "ttgamma_inclusive"
INCLUSIVE_TTBAR_SAMPLE = "inclusive_ttbar"
OTHER_SAMPLE = "other"

SPLIT_SAMPLE_ROLE_POLICY = "split"
RUN2_NLO_INCLUSIVE_SAMPLE_ROLE_POLICY = "run2_nlo_inclusive"
SUPPORTED_SAMPLE_ROLE_POLICIES = (
    SPLIT_SAMPLE_ROLE_POLICY,
    RUN2_NLO_INCLUSIVE_SAMPLE_ROLE_POLICY,
)

_RUN2_ERA = r"(?:UL16APV|UL16|UL17|UL18)"
_RUN3_ERA = r"(?:2022EE|2022|2023BPix|2023)"
_TTGAMMA_PRODUCTION_SAMPLE_PATTERN = re.compile(
    rf"(?:{_RUN2_ERA}_TTGJets(?:_NDSkim)?|TTGJets_central{_RUN2_ERA})"
)
_TTGAMMA_DECAY_SAMPLE_PATTERN = re.compile(
    rf"(?:{_RUN2_ERA}_TTGamma_(?:Dilept|SingleLept)(?:_NDSkim)?"
    rf"|TTGamma_central{_RUN2_ERA})"
)
_TTGAMMA_INCLUSIVE_SAMPLE_PATTERN = re.compile(
    rf"(?:TTG-1Jets_PTG-(?:10to100|100to200|200)"
    rf"(?:_NDSkim)?_{_RUN3_ERA}"
    rf"|TTG-1Jets_PTG-(?:10to100|100to200|200)_central{_RUN3_ERA}"
    rf"|TTGamma_central{_RUN3_ERA})"
)
_INCLUSIVE_TTBAR_SAMPLE_PATTERN = re.compile(
    rf"(?:{_RUN2_ERA}_(?:TTTo2L2Nu|TTToSemiLeptonic|TTToHadronic|TTJets)"
    rf"(?:_NDSkim)?"
    rf"|(?:TTTo2L2Nu|TTToSemiLeptonic|TTToHadronic|TTJets)"
    rf"_central{_RUN2_ERA}"
    rf"|(?:TTto2L2Nu(?:-[23]Jets)?|TTtoLNu2Q|TTto4Q)"
    rf"(?:_NDSkim)?_{_RUN3_ERA}"
    rf"|(?:TTto2L2Nu|TTtoLNu2Q|TTto4Q)_central{_RUN3_ERA})"
)


def get_ttgamma_sample_role_policy(
    sample_role_policy=SPLIT_SAMPLE_ROLE_POLICY,
):
    """Return the configured ttgamma sample-role policy."""

    policy = sample_role_policy
    if policy not in SUPPORTED_SAMPLE_ROLE_POLICIES:
        supported = ", ".join(SUPPORTED_SAMPLE_ROLE_POLICIES)
        raise ValueError(
            f"Unsupported ttgamma sample-role policy {policy!r}; "
            f"supported values are: {supported}"
        )
    return policy


def _zeros_like_objects(selected_leptons, dtype):
    return ak.values_astype(
        ak.zeros_like(ak.local_index(selected_leptons, axis=1)),
        dtype,
    )


def _full_like_objects(selected_leptons, value, dtype=np.int64):
    return _zeros_like_objects(selected_leptons, dtype) + value


def _zeros_like_events(selected_leptons, dtype):
    return ak.values_astype(
        ak.zeros_like(ak.num(selected_leptons, axis=1)),
        dtype,
    )


def _full_like_events(selected_leptons, value, dtype=np.bool_):
    return _zeros_like_events(selected_leptons, dtype) + value


def classify_conversion_overlap_sample(
    sample_name,
    sample_role_policy=SPLIT_SAMPLE_ROLE_POLICY,
):
    """Classify a dataset-basename key for conversion overlap removal."""

    normalized_name = str(sample_name).rsplit("/", 1)[-1]
    if normalized_name.endswith(".json"):
        normalized_name = normalized_name[:-5]
    policy = get_ttgamma_sample_role_policy(sample_role_policy)
    if (
        policy == RUN2_NLO_INCLUSIVE_SAMPLE_ROLE_POLICY
        and _TTGAMMA_PRODUCTION_SAMPLE_PATTERN.fullmatch(normalized_name)
    ):
        return TTGAMMA_INCLUSIVE_SAMPLE
    if _TTGAMMA_PRODUCTION_SAMPLE_PATTERN.fullmatch(normalized_name):
        return TTGAMMA_PRODUCTION_SAMPLE
    if _TTGAMMA_DECAY_SAMPLE_PATTERN.fullmatch(normalized_name):
        return TTGAMMA_DECAY_SAMPLE
    if _TTGAMMA_INCLUSIVE_SAMPLE_PATTERN.fullmatch(normalized_name):
        return TTGAMMA_INCLUSIVE_SAMPLE
    if _INCLUSIVE_TTBAR_SAMPLE_PATTERN.fullmatch(normalized_name):
        return INCLUSIVE_TTBAR_SAMPLE
    return OTHER_SAMPLE


def attach_conversion_overlap_removal_diagnostics(
    events,
    sample_name,
    is_data=False,
    sample_role_policy=SPLIT_SAMPLE_ROLE_POLICY,
):
    """Attach nominal ttgamma/ttbar conversion-overlap removal flags."""

    decay = ak.values_astype(
        ak.fill_none(
            events[
                f"{EVENT_DIAGNOSTIC_PREFIX}"
                "has_decay_origin_conversion_photon"
            ],
            False,
        ),
        np.bool_,
    )
    production = ak.values_astype(
        ak.fill_none(
            events[
                f"{EVENT_DIAGNOSTIC_PREFIX}"
                "has_production_origin_conversion_photon"
            ],
            False,
        ),
        np.bool_,
    )
    has_selected_conversion_lepton = ak.values_astype(
        ak.fill_none(
            events[
                f"{EVENT_DIAGNOSTIC_PREFIX}"
                "has_selected_conversion_lepton"
            ],
            False,
        ),
        np.bool_,
    )
    sample_class = (
        OTHER_SAMPLE
        if is_data
        else classify_conversion_overlap_sample(
            sample_name,
            sample_role_policy=sample_role_policy,
        )
    )

    false_events = ak.zeros_like(decay, dtype=np.bool_)
    true_events = ~false_events
    is_ttgamma_production = (
        true_events
        if sample_class == TTGAMMA_PRODUCTION_SAMPLE
        else false_events
    )
    is_ttgamma_decay = (
        true_events
        if sample_class == TTGAMMA_DECAY_SAMPLE
        else false_events
    )
    is_ttgamma_inclusive = (
        true_events
        if sample_class == TTGAMMA_INCLUSIVE_SAMPLE
        else false_events
    )
    is_inclusive_ttbar = (
        true_events
        if sample_class == INCLUSIVE_TTBAR_SAMPLE
        else false_events
    )
    removed_ttgamma_production_role_decay_origin = (
        decay if sample_class == TTGAMMA_PRODUCTION_SAMPLE else false_events
    )
    removed_ttgamma_decay_role_production_origin = (
        production if sample_class == TTGAMMA_DECAY_SAMPLE else false_events
    )
    removed_ttbar_selected_external_conversion = (
        has_selected_conversion_lepton
        if sample_class == INCLUSIVE_TTBAR_SAMPLE
        else false_events
    )
    removed = (
        removed_ttgamma_production_role_decay_origin
        | removed_ttgamma_decay_role_production_origin
        | removed_ttbar_selected_external_conversion
    )
    result = {
        "pass_conversion_overlap_removal": ~removed,
        "removed_by_conversion_overlap_removal": removed,
        "removed_ttgamma_production_role_decay_origin": (
            removed_ttgamma_production_role_decay_origin
        ),
        "removed_ttgamma_decay_role_production_origin": (
            removed_ttgamma_decay_role_production_origin
        ),
        "removed_ttbar_selected_external_conversion": (
            removed_ttbar_selected_external_conversion
        ),
        "sample_role_is_ttgamma_production": is_ttgamma_production,
        "sample_role_is_ttgamma_decay": is_ttgamma_decay,
        "sample_role_is_ttgamma_inclusive": is_ttgamma_inclusive,
        "sample_role_is_inclusive_ttbar": is_inclusive_ttbar,
        "has_mixed_decay_and_production_conversion_photons": (
            decay & production
        ),
    }
    for field, values in result.items():
        events[f"{EVENT_DIAGNOSTIC_PREFIX}{field}"] = values
    return result


def _has_required_fields(genparts, selected_leptons):
    if genparts is None or selected_leptons is None:
        return False
    return (
        {"pdgId", "genPartIdxMother"} <= set(ak.fields(genparts))
        and {"genPartFlav", "genPartIdx"} <= set(ak.fields(selected_leptons))
    )


def _broadcast_genpart_count(genparts, indices):
    return ak.broadcast_arrays(ak.num(genparts, axis=1), indices)[0]


def _valid_index(genparts, indices):
    return (indices >= 0) & (indices < _broadcast_genpart_count(genparts, indices))


def _take_genpart_field(genparts, indices, field, default):
    valid = _valid_index(genparts, indices)
    masked_indices = ak.mask(indices, valid)
    return ak.fill_none(genparts[masked_indices][field], default)


def _category_mask(categories, accepted):
    mask = ak.zeros_like(categories, dtype=np.bool_)
    for category in accepted:
        mask = mask | (categories == category)
    return mask


def _repeated_index(history, current, active):
    if history is None:
        return ak.zeros_like(active)
    return active & ak.any(
        history == current,
        axis=2,
    )


def _append_history(history, current):
    current_step = ak.singletons(current)
    if history is None:
        return current_step
    return ak.concatenate([history, current_step], axis=2)


def _empty_result(selected_leptons, missing_branches):
    categories = _full_like_objects(selected_leptons, NOT_CONVERSION)
    indices = _full_like_objects(selected_leptons, -1)
    false_objects = _zeros_like_objects(selected_leptons, np.bool_)
    false_events = _zeros_like_events(selected_leptons, np.bool_)
    zero_counts = _zeros_like_events(selected_leptons, np.int64)

    return {
        "lepton": {
            "category": categories,
            "recovered_photon_index": indices,
            "first_copy_photon_index": indices,
            "is_selected_conversion_lepton": false_objects,
            "has_recovered_conversion_photon": false_objects,
        },
        "event": {
            "diagnostic_missing_branches": _full_like_events(
                selected_leptons, missing_branches
            ),
            "has_selected_conversion_lepton": false_events,
            "has_recovered_conversion_photon": false_events,
            "has_classified_origin_conversion_photon": false_events,
            "has_decay_origin_conversion_photon": false_events,
            "has_production_origin_conversion_photon": false_events,
            "has_hadron_ancestor_conversion_photon": false_events,
            "has_ambiguous_conversion_photon": false_events,
            "has_no_photon_found_conversion_lepton": false_events,
            "has_invalid_match_conversion_lepton": false_events,
            "n_selected_conversion_leptons": zero_counts,
            "n_recovered_conversion_photons": zero_counts,
            "n_classified_origin_conversion_photons": zero_counts,
            "n_decay_origin_conversion_photons": zero_counts,
            "n_production_origin_conversion_photons": zero_counts,
            "n_hadron_ancestor_conversion_photons": zero_counts,
            "n_ambiguous_conversion_photons": zero_counts,
            "n_no_photon_found_conversion_leptons": zero_counts,
            "n_invalid_match_conversion_leptons": zero_counts,
        },
    }


def _recover_photon_indices(genparts, selected_leptons, max_depth):
    is_conversion = selected_leptons.genPartFlav == 22
    initial_indices = selected_leptons.genPartIdx
    initial_valid = _valid_index(genparts, initial_indices)

    current = ak.where(initial_valid, initial_indices, -1)
    found = _full_like_objects(selected_leptons, -1)
    malformed = is_conversion & ~initial_valid
    active = is_conversion & initial_valid
    history = None

    for _ in range(max_depth):
        if not bool(ak.any(active)):
            break
        repeated = _repeated_index(history, current, active)
        malformed = malformed | repeated
        active = active & ~repeated

        valid = _valid_index(genparts, current)
        invalid = active & (current != -1) & ~valid
        malformed = malformed | invalid
        active = active & valid

        pdg_id = _take_genpart_field(genparts, current, "pdgId", 0)
        found_now = active & (abs(pdg_id) == 22)
        found = ak.where(found_now, current, found)
        active = active & ~found_now

        history = _append_history(history, current)
        mother = _take_genpart_field(
            genparts, current, "genPartIdxMother", -1
        )
        active = active & (mother != -1)
        current = mother

    malformed = malformed | active
    return is_conversion, found, malformed


def _first_photon_copy(genparts, photon_indices, max_depth):
    first_copy = photon_indices
    active = photon_indices >= 0
    malformed = ak.zeros_like(active)
    history = None

    for _ in range(max_depth):
        if not bool(ak.any(active)):
            break
        repeated = _repeated_index(history, first_copy, active)
        malformed = malformed | repeated
        active = active & ~repeated

        mother = _take_genpart_field(
            genparts, first_copy, "genPartIdxMother", -1
        )
        mother_valid = _valid_index(genparts, mother)
        invalid = active & (mother != -1) & ~mother_valid
        malformed = malformed | invalid

        mother_pdg_id = _take_genpart_field(genparts, mother, "pdgId", 0)
        same_pdg_mother = active & mother_valid & (abs(mother_pdg_id) == 22)
        history = _append_history(history, first_copy)
        first_copy = ak.where(same_pdg_mother, mother, first_copy)
        active = same_pdg_mother

    malformed = malformed | active
    return first_copy, malformed


def _photon_ancestry(genparts, photon_indices, max_depth):
    mother = _take_genpart_field(
        genparts, photon_indices, "genPartIdxMother", -1
    )
    mother_valid = _valid_index(genparts, mother)
    mother_pdg_id = _take_genpart_field(genparts, mother, "pdgId", 0)
    grandmother = _take_genpart_field(
        genparts, mother, "genPartIdxMother", -1
    )
    grandmother_pdg_id = _take_genpart_field(
        genparts, grandmother, "pdgId", 0
    )

    active = (photon_indices >= 0) & mother_valid
    malformed = (photon_indices >= 0) & (mother != -1) & ~mother_valid
    has_top = ak.zeros_like(active)
    has_hadron = ak.zeros_like(active)
    current = mother
    history = None

    for _ in range(max_depth):
        if not bool(ak.any(active)):
            break
        repeated = _repeated_index(history, current, active)
        malformed = malformed | repeated
        active = active & ~repeated

        valid = _valid_index(genparts, current)
        invalid = active & (current != -1) & ~valid
        malformed = malformed | invalid
        active = active & valid

        pdg_id = abs(_take_genpart_field(genparts, current, "pdgId", 0))
        has_top = has_top | (active & (pdg_id == 6))
        has_hadron = has_hadron | (
            active & (pdg_id > 37) & (pdg_id != 2212)
        )

        history = _append_history(history, current)
        next_mother = _take_genpart_field(
            genparts, current, "genPartIdxMother", -1
        )
        active = active & (next_mother != -1)
        current = next_mother

    malformed = malformed | active
    no_mother = (photon_indices >= 0) & (mother == -1)
    return {
        "mother_pdg_id": mother_pdg_id,
        "grandmother_pdg_id": grandmother_pdg_id,
        "has_top": has_top,
        "has_hadron": has_hadron,
        "malformed": malformed,
        "no_mother": no_mother,
    }


def _classify_photons(
    genparts,
    is_conversion,
    recovered_photon_indices,
    recovery_malformed,
    max_depth,
):
    categories = _full_like_objects(is_conversion, NOT_CONVERSION)
    no_match = is_conversion & (recovered_photon_indices < 0)
    categories = ak.where(
        no_match & recovery_malformed, INVALID_MATCH, categories
    )
    categories = ak.where(
        no_match & ~recovery_malformed, NO_PHOTON_FOUND, categories
    )

    first_copy, first_copy_malformed = _first_photon_copy(
        genparts, recovered_photon_indices, max_depth
    )
    ancestry = _photon_ancestry(genparts, first_copy, max_depth)
    recovered = is_conversion & (recovered_photon_indices >= 0)
    ambiguous = recovered & (
        first_copy_malformed
        | ancestry["malformed"]
        | ancestry["no_mother"]
    )

    mother_abs = abs(ancestry["mother_pdg_id"])
    decay_lepton = recovered & (
        (mother_abs == 11) | (mother_abs == 13) | (mother_abs == 15)
    )
    decay_w_or_b = (
        recovered
        & ((mother_abs == 24) | (mother_abs == 5))
        & ancestry["has_top"]
    )
    decay_top_copy = (
        recovered
        & (mother_abs == 6)
        & (
            ancestry["grandmother_pdg_id"]
            == ancestry["mother_pdg_id"]
        )
    )
    offshell_top = recovered & (
        ((mother_abs == 6) & ~decay_top_copy) | (mother_abs == 21)
    )
    decay_any = decay_lepton | decay_w_or_b | decay_top_copy
    production_isr = recovered & ~decay_any & ~offshell_top

    categories = ak.where(production_isr, PRODUCTION_ISR, categories)
    categories = ak.where(offshell_top, PRODUCTION_OFFSHELL_TOP, categories)
    categories = ak.where(
        decay_top_copy, DECAY_TOP_COPY_CONDITION, categories
    )
    categories = ak.where(
        decay_w_or_b, DECAY_W_OR_B_WITH_TOP_ANCESTOR, categories
    )
    categories = ak.where(decay_lepton, DECAY_LEPTON, categories)
    categories = ak.where(
        recovered & ancestry["has_hadron"], HADRON_ANCESTOR, categories
    )
    categories = ak.where(ambiguous, AMBIGUOUS, categories)
    return categories, first_copy


def _event_reduction(selected_leptons, categories, missing_branches=False):
    is_conversion = selected_leptons.genPartFlav == 22
    decay = _category_mask(categories, DECAY_CATEGORIES)
    production = _category_mask(categories, PRODUCTION_CATEGORIES)
    classified_origin = decay | production
    hadron = categories == HADRON_ANCESTOR
    ambiguous = categories == AMBIGUOUS
    recovered = classified_origin | hadron | ambiguous
    no_photon = categories == NO_PHOTON_FOUND
    invalid = categories == INVALID_MATCH

    return {
        "diagnostic_missing_branches": _full_like_events(
            selected_leptons, missing_branches
        ),
        "has_selected_conversion_lepton": ak.any(is_conversion, axis=1),
        "has_recovered_conversion_photon": ak.any(recovered, axis=1),
        "has_classified_origin_conversion_photon": ak.any(
            classified_origin, axis=1
        ),
        "has_decay_origin_conversion_photon": ak.any(decay, axis=1),
        "has_production_origin_conversion_photon": ak.any(
            production, axis=1
        ),
        "has_hadron_ancestor_conversion_photon": ak.any(hadron, axis=1),
        "has_ambiguous_conversion_photon": ak.any(ambiguous, axis=1),
        "has_no_photon_found_conversion_lepton": ak.any(no_photon, axis=1),
        "has_invalid_match_conversion_lepton": ak.any(invalid, axis=1),
        "n_selected_conversion_leptons": ak.sum(is_conversion, axis=1),
        "n_recovered_conversion_photons": ak.sum(recovered, axis=1),
        "n_classified_origin_conversion_photons": ak.sum(
            classified_origin, axis=1
        ),
        "n_decay_origin_conversion_photons": ak.sum(decay, axis=1),
        "n_production_origin_conversion_photons": ak.sum(
            production, axis=1
        ),
        "n_hadron_ancestor_conversion_photons": ak.sum(hadron, axis=1),
        "n_ambiguous_conversion_photons": ak.sum(ambiguous, axis=1),
        "n_no_photon_found_conversion_leptons": ak.sum(no_photon, axis=1),
        "n_invalid_match_conversion_leptons": ak.sum(invalid, axis=1),
    }


def classify_selected_conversion_photon_history(
    genparts,
    selected_leptons,
    max_depth=DEFAULT_MAX_ANCESTRY_DEPTH,
):
    """Classify gen-photon history for selected FO conversion-like leptons.

    The result contains jagged lepton-level arrays and flat event-level
    diagnostic reductions. It does not make or apply an event-selection
    decision.
    """

    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if selected_leptons is None:
        raise ValueError("selected_leptons is required")
    if not _has_required_fields(genparts, selected_leptons):
        return _empty_result(selected_leptons, missing_branches=True)

    is_conversion, recovered_photon_indices, recovery_malformed = (
        _recover_photon_indices(genparts, selected_leptons, max_depth)
    )
    categories, first_copy_indices = _classify_photons(
        genparts,
        is_conversion,
        recovered_photon_indices,
        recovery_malformed,
        max_depth,
    )
    has_recovered_photon = is_conversion & (recovered_photon_indices >= 0)

    return {
        "lepton": {
            "category": categories,
            "recovered_photon_index": recovered_photon_indices,
            "first_copy_photon_index": first_copy_indices,
            "is_selected_conversion_lepton": is_conversion,
            "has_recovered_conversion_photon": has_recovered_photon,
        },
        "event": _event_reduction(selected_leptons, categories),
    }


def attach_photon_history_diagnostics(events, selected_leptons, genparts=None):
    """Attach photon-history fields and return the decorated leptons."""

    result = classify_selected_conversion_photon_history(
        genparts, selected_leptons
    )
    leptons = selected_leptons
    for field, values in result["lepton"].items():
        leptons = ak.with_field(
            leptons, values, f"conversion_photon_history_{field}"
        )
    for field, values in result["event"].items():
        events[f"{EVENT_DIAGNOSTIC_PREFIX}{field}"] = values
    return leptons
