from pathlib import Path

import awkward as ak
import pytest

from topeft.modules.ttgamma_photon_history import (
    AMBIGUOUS,
    DECAY_LEPTON,
    DECAY_TOP_COPY_CONDITION,
    DECAY_W_OR_B_WITH_TOP_ANCESTOR,
    HADRON_ANCESTOR,
    INVALID_MATCH,
    NO_PHOTON_FOUND,
    NOT_CONVERSION,
    PRODUCTION_ISR,
    PRODUCTION_OFFSHELL_TOP,
    attach_conversion_overlap_removal_diagnostics,
    attach_photon_history_diagnostics,
    classify_conversion_overlap_sample,
    classify_selected_conversion_photon_history,
)
from topeft.modules.yield_tools import YieldTools


REPO_ROOT = Path(__file__).resolve().parents[1]


def _genpart(pdg_id, mother=-1):
    return {"pdgId": pdg_id, "genPartIdxMother": mother}


def _lepton(gen_part_idx, gen_part_flav=22):
    return {"genPartIdx": gen_part_idx, "genPartFlav": gen_part_flav}


def _classify(genparts, leptons, max_depth=64):
    return classify_selected_conversion_photon_history(
        ak.Array(genparts),
        ak.Array(leptons),
        max_depth=max_depth,
    )


def _categories(result):
    return ak.to_list(result["lepton"]["category"])


def _event(result, field):
    return ak.to_list(result["event"][field])


def _overlap_events(decay, production, has_selected=None, **guard_fields):
    if has_selected is None:
        has_selected = [
            has_decay or has_production
            for has_decay, has_production in zip(decay, production)
        ]
    events = {
        "ttgamma_photon_history_has_selected_conversion_lepton": ak.Array(
            has_selected
        ),
        "ttgamma_photon_history_has_decay_origin_conversion_photon": ak.Array(
            decay
        ),
        "ttgamma_photon_history_has_production_origin_conversion_photon": (
            ak.Array(production)
        ),
    }
    for field, values in guard_fields.items():
        events[f"ttgamma_photon_history_{field}"] = ak.Array(values)
    return events


def _attach_overlap(
    sample_name,
    decay,
    production,
    has_selected=None,
    is_data=False,
    sample_role_policy="split",
    **guards,
):
    events = _overlap_events(
        decay,
        production,
        has_selected=has_selected,
        **guards,
    )
    result = attach_conversion_overlap_removal_diagnostics(
        events,
        sample_name=sample_name,
        is_data=is_data,
        sample_role_policy=sample_role_policy,
    )
    return events, {
        field: ak.to_list(values) for field, values in result.items()
    }


@pytest.mark.parametrize(
    ("sample_name", "expected"),
    [
        ("TTGJets_centralUL18", "ttgamma_production"),
        ("UL18_TTGJets_NDSkim", "ttgamma_production"),
        ("TTGamma_centralUL18", "ttgamma_decay"),
        ("UL18_TTGamma_Dilept_NDSkim", "ttgamma_decay"),
        ("TTG-1Jets_PTG-10to100_NDSkim_2022EE", "ttgamma_inclusive"),
        ("TTto2L2Nu_central2022EE", "inclusive_ttbar"),
    ],
)
def test_default_sample_role_policy_is_split(sample_name, expected):
    assert classify_conversion_overlap_sample(sample_name) == expected


def test_explicit_split_sample_role_policy_keeps_run2_ttgamma_nlo_split():
    assert (
        classify_conversion_overlap_sample(
            "TTGJets_centralUL18",
            sample_role_policy="split",
        )
        == "ttgamma_production"
    )
    assert (
        classify_conversion_overlap_sample(
            "UL18_TTGJets_NDSkim",
            sample_role_policy="split",
        )
        == "ttgamma_production"
    )


@pytest.mark.parametrize(
    ("sample_name", "expected"),
    [
        ("TTGJets_centralUL18", "ttgamma_inclusive"),
        ("UL18_TTGJets_NDSkim", "ttgamma_inclusive"),
        ("TTGamma_centralUL18", "ttgamma_decay"),
        ("TTG-1Jets_PTG-10to100_NDSkim_2022EE", "ttgamma_inclusive"),
        ("TTTo2L2Nu_centralUL18", "inclusive_ttbar"),
    ],
)
def test_run2_nlo_inclusive_policy_classification(sample_name, expected):
    assert (
        classify_conversion_overlap_sample(
            sample_name,
            sample_role_policy="run2_nlo_inclusive",
        )
        == expected
    )


@pytest.mark.parametrize(
    ("sample_name", "expected"),
    [
        ("TTGamma_centralUL18", "ttgamma_decay"),
        ("UL18_TTGamma_Dilept_NDSkim", "ttgamma_decay"),
        (
            "TTG-1Jets_PTG-10to100_NDSkim_2022",
            "ttgamma_inclusive",
        ),
        ("TTTo2L2Nu_centralUL18", "inclusive_ttbar"),
    ],
)
def test_run2_nlo_inclusive_policy_preserves_other_roles(
    sample_name,
    expected,
):
    assert (
        classify_conversion_overlap_sample(
            sample_name,
            sample_role_policy="run2_nlo_inclusive",
        )
        == expected
    )


@pytest.mark.parametrize(
    ("decay", "production"),
    [([True], [False]), ([False], [True])],
)
def test_run2_nlo_inclusive_policy_keeps_run2_ttgamma_nlo_origins(
    decay,
    production,
):
    _, result = _attach_overlap(
        "UL18_TTGJets_NDSkim",
        decay,
        production,
        sample_role_policy="run2_nlo_inclusive",
    )

    assert result["sample_role_is_ttgamma_inclusive"] == [True]
    assert result["removed_by_conversion_overlap_removal"] == [False]
    assert result["pass_conversion_overlap_removal"] == [True]


def test_run2_nlo_inclusive_policy_keeps_inclusive_ttbar_selected_veto():
    _, result = _attach_overlap(
        "UL18_TTTo2L2Nu_NDSkim",
        [False],
        [False],
        has_selected=[True],
        sample_role_policy="run2_nlo_inclusive",
    )

    assert result["sample_role_is_inclusive_ttbar"] == [True]
    assert result["removed_ttbar_selected_external_conversion"] == [True]
    assert result["removed_by_conversion_overlap_removal"] == [True]
    assert result["pass_conversion_overlap_removal"] == [False]


def test_invalid_sample_role_policy_raises():
    with pytest.raises(ValueError, match="unsupported_policy"):
        classify_conversion_overlap_sample(
            "UL18_TTGJets_NDSkim",
            sample_role_policy="unsupported_policy",
        )


@pytest.mark.parametrize(
    ("sample_name", "expected"),
    [
        ("UL16APV_TTGJets_NDSkim", "ttgamma_production"),
        ("UL16_TTGJets_NDSkim", "ttgamma_production"),
        ("UL17_TTGJets_NDSkim", "ttgamma_production"),
        ("UL18_TTGJets_NDSkim", "ttgamma_production"),
        ("TTGJets_centralUL18", "ttgamma_production"),
        ("UL16APV_TTGamma_Dilept_NDSkim", "ttgamma_decay"),
        ("UL16APV_TTGamma_SingleLept_NDSkim", "ttgamma_decay"),
        ("UL16_TTGamma_Dilept_NDSkim", "ttgamma_decay"),
        ("UL16_TTGamma_SingleLept_NDSkim", "ttgamma_decay"),
        ("UL17_TTGamma_Dilept_NDSkim", "ttgamma_decay"),
        ("UL17_TTGamma_SingleLept_NDSkim", "ttgamma_decay"),
        ("UL18_TTGamma_Dilept_NDSkim", "ttgamma_decay"),
        ("UL18_TTGamma_SingleLept_NDSkim", "ttgamma_decay"),
        ("TTGamma_centralUL18", "ttgamma_decay"),
        (
            "TTG-1Jets_PTG-10to100_NDSkim_2022",
            "ttgamma_inclusive",
        ),
        (
            "TTG-1Jets_PTG-100to200_NDSkim_2022",
            "ttgamma_inclusive",
        ),
        (
            "TTG-1Jets_PTG-200_NDSkim_2022",
            "ttgamma_inclusive",
        ),
        (
            "TTG-1Jets_PTG-10to100_NDSkim_2022EE",
            "ttgamma_inclusive",
        ),
        (
            "TTG-1Jets_PTG-100to200_NDSkim_2022EE",
            "ttgamma_inclusive",
        ),
        (
            "TTG-1Jets_PTG-200_NDSkim_2022EE",
            "ttgamma_inclusive",
        ),
        (
            "TTG-1Jets_PTG-10to100_NDSkim_2023",
            "ttgamma_inclusive",
        ),
        (
            "TTG-1Jets_PTG-100to200_NDSkim_2023",
            "ttgamma_inclusive",
        ),
        (
            "TTG-1Jets_PTG-200_NDSkim_2023",
            "ttgamma_inclusive",
        ),
        (
            "TTG-1Jets_PTG-10to100_NDSkim_2023BPix",
            "ttgamma_inclusive",
        ),
        (
            "TTG-1Jets_PTG-100to200_NDSkim_2023BPix",
            "ttgamma_inclusive",
        ),
        (
            "TTG-1Jets_PTG-200_NDSkim_2023BPix",
            "ttgamma_inclusive",
        ),
        (
            "TTG-1Jets_PTG-100to200_central2022",
            "ttgamma_inclusive",
        ),
        ("TTGamma_central2022EE", "ttgamma_inclusive"),
        ("UL16APV_TTTo2L2Nu_NDSkim", "inclusive_ttbar"),
        ("UL16APV_TTToSemiLeptonic_NDSkim", "inclusive_ttbar"),
        ("UL16_TTTo2L2Nu_NDSkim", "inclusive_ttbar"),
        ("UL16_TTToSemiLeptonic_NDSkim", "inclusive_ttbar"),
        ("UL17_TTTo2L2Nu_NDSkim", "inclusive_ttbar"),
        ("UL17_TTToSemiLeptonic_NDSkim", "inclusive_ttbar"),
        ("UL18_TTTo2L2Nu_NDSkim", "inclusive_ttbar"),
        ("UL18_TTToSemiLeptonic_NDSkim", "inclusive_ttbar"),
        ("UL18_TTToHadronic", "inclusive_ttbar"),
        ("UL18_TTJets", "inclusive_ttbar"),
        ("TTto2L2Nu-2Jets_2022", "inclusive_ttbar"),
        ("TTto2L2Nu_NDSkim_2022", "inclusive_ttbar"),
        ("TTtoLNu2Q_NDSkim_2022", "inclusive_ttbar"),
        ("TTto2L2Nu_NDSkim_2022EE", "inclusive_ttbar"),
        ("TTtoLNu2Q_NDSkim_2022EE", "inclusive_ttbar"),
        ("TTto2L2Nu_NDSkim_2023", "inclusive_ttbar"),
        ("TTtoLNu2Q_NDSkim_2023", "inclusive_ttbar"),
        ("TTto2L2Nu_NDSkim_2023BPix", "inclusive_ttbar"),
        ("TTtoLNu2Q_NDSkim_2023BPix", "inclusive_ttbar"),
        ("TTto4Q_2023BPix", "inclusive_ttbar"),
        ("TTto2L2Nu_central2022EE", "inclusive_ttbar"),
        ("input/Muon_2022.json", "other"),
    ],
)
def test_conversion_overlap_sample_classifier_supported_names(
    sample_name, expected
):
    assert classify_conversion_overlap_sample(sample_name) == expected


@pytest.mark.parametrize(
    "sample_name",
    [
        "TTWJetsToLNu",
        "TTZToLLNuNu_M_10",
        "ttHnobb",
        "ST_top_t-channel_2022",
        "ST_tW_Leptonic_2023",
        "ZZTo4l_TTJets",
        "Muon_2022",
        "TTGamma_unknown",
        "TTG-1Jets_PTG-10to100_2018",
    ],
)
def test_conversion_overlap_sample_classifier_is_conservative(sample_name):
    assert classify_conversion_overlap_sample(sample_name) == "other"


@pytest.mark.parametrize(
    "year",
    ["2022", "2022EE", "2023", "2023BPix"],
)
@pytest.mark.parametrize(
    "pt_bin",
    ["10to100", "100to200", "200"],
)
def test_run3_ttgamma_hist_axes_are_in_conversion_group(year, pt_bin):
    process = f"TTG-1Jets_PTG-{pt_bin}_central{year}"
    assert YieldTools().get_short_name(process) == "conv"


@pytest.mark.parametrize(
    "sample_name",
    [
        "UL16APV_TTGJets_NDSkim",
        "UL16_TTGJets_NDSkim",
        "UL17_TTGJets_NDSkim",
        "UL18_TTGJets_NDSkim",
    ],
)
def test_ttgamma_production_role_removes_decay_origin_only(sample_name):
    _, result = _attach_overlap(sample_name, [True], [False])

    assert result[
        "removed_ttgamma_production_role_decay_origin"
    ] == [True]
    assert result[
        "removed_ttgamma_decay_role_production_origin"
    ] == [False]
    assert result["removed_ttbar_selected_external_conversion"] == [False]
    assert result["removed_by_conversion_overlap_removal"] == [True]
    assert result["pass_conversion_overlap_removal"] == [False]


def test_ttgamma_production_role_keeps_production_origin():
    _, result = _attach_overlap("UL18_TTGJets_NDSkim", [False], [True])

    assert result["removed_by_conversion_overlap_removal"] == [False]
    assert result["pass_conversion_overlap_removal"] == [True]


@pytest.mark.parametrize(
    "sample_name",
    [
        "UL16APV_TTGamma_Dilept_NDSkim",
        "UL16_TTGamma_SingleLept_NDSkim",
        "UL17_TTGamma_Dilept_NDSkim",
        "UL18_TTGamma_SingleLept_NDSkim",
    ],
)
def test_ttgamma_decay_role_removes_production_origin_only(sample_name):
    _, result = _attach_overlap(sample_name, [False], [True])

    assert result[
        "removed_ttgamma_production_role_decay_origin"
    ] == [False]
    assert result[
        "removed_ttgamma_decay_role_production_origin"
    ] == [True]
    assert result["removed_ttbar_selected_external_conversion"] == [False]
    assert result["removed_by_conversion_overlap_removal"] == [True]
    assert result["pass_conversion_overlap_removal"] == [False]


def test_ttgamma_decay_role_keeps_decay_origin():
    _, result = _attach_overlap(
        "UL18_TTGamma_Dilept_NDSkim",
        [True],
        [False],
    )

    assert result["removed_by_conversion_overlap_removal"] == [False]
    assert result["pass_conversion_overlap_removal"] == [True]


@pytest.mark.parametrize(
    "sample_name",
    [
        "TTG-1Jets_PTG-10to100_NDSkim_2022",
        "TTG-1Jets_PTG-100to200_NDSkim_2022EE",
        "TTG-1Jets_PTG-200_NDSkim_2023",
        "TTG-1Jets_PTG-10to100_NDSkim_2023BPix",
    ],
)
@pytest.mark.parametrize(
    ("decay", "production"),
    [([True], [False]), ([False], [True]), ([True], [True])],
)
def test_ttgamma_inclusive_keeps_all_classified_origins(
    sample_name,
    decay,
    production,
):
    _, result = _attach_overlap(sample_name, decay, production)

    assert result["removed_by_conversion_overlap_removal"] == [False]
    assert result["pass_conversion_overlap_removal"] == [True]


@pytest.mark.parametrize(
    "sample_name",
    [
        "UL18_TTTo2L2Nu",
        "UL18_TTToSemiLeptonic",
        "UL18_TTToHadronic",
        "UL18_TTJets",
        "TTto2L2Nu_2022",
        "TTtoLNu2Q_2023",
        "TTto4Q_2023BPix",
    ],
)
def test_inclusive_ttbar_removes_selected_external_conversion(sample_name):
    _, result = _attach_overlap(
        sample_name,
        [False],
        [False],
        has_selected=[True],
    )

    assert result["removed_ttbar_selected_external_conversion"] == [True]
    assert result["removed_by_conversion_overlap_removal"] == [True]
    assert result["pass_conversion_overlap_removal"] == [False]


def test_inclusive_ttbar_without_selected_conversion_passes():
    _, result = _attach_overlap(
        "UL18_TTTo2L2Nu",
        [True],
        [True],
        has_selected=[False],
    )

    assert result["removed_by_conversion_overlap_removal"] == [False]
    assert result["pass_conversion_overlap_removal"] == [True]


@pytest.mark.parametrize(
    "sample_name",
    ["UL18_TTGJets", "UL18_TTGamma_Dilept"],
)
def test_ttgamma_guard_categories_do_not_drive_role_veto(sample_name):
    _, result = _attach_overlap(
        sample_name,
        [False],
        [False],
        has_recovered_conversion_photon=[True],
        has_classified_origin_conversion_photon=[False],
        has_hadron_ancestor_conversion_photon=[True],
        has_ambiguous_conversion_photon=[True],
        has_no_photon_found_conversion_lepton=[True],
        has_invalid_match_conversion_lepton=[True],
    )

    assert result["removed_by_conversion_overlap_removal"] == [False]
    assert result["pass_conversion_overlap_removal"] == [True]


def test_inclusive_ttbar_veto_does_not_require_recovered_photon_ancestry():
    _, result = _attach_overlap(
        "UL18_TTTo2L2Nu",
        [False],
        [False],
        has_selected=[True],
        has_recovered_conversion_photon=[False],
        has_classified_origin_conversion_photon=[False],
        has_no_photon_found_conversion_lepton=[True],
        has_invalid_match_conversion_lepton=[True],
    )

    assert result["removed_ttbar_selected_external_conversion"] == [True]
    assert result["pass_conversion_overlap_removal"] == [False]


@pytest.mark.parametrize(
    ("sample_name", "is_data", "expected_pass"),
    [
        ("WJetsToLNu_2022", False, True),
        ("UL18_TTGamma_Dilept", True, True),
        ("UL18_TTTo2L2Nu", True, True),
    ],
)
def test_other_mc_and_data_pass_unchanged(
    sample_name, is_data, expected_pass
):
    _, result = _attach_overlap(
        sample_name,
        [True],
        [True],
        is_data=is_data,
    )

    assert result[
        "removed_ttgamma_production_role_decay_origin"
    ] == [False]
    assert result[
        "removed_ttgamma_decay_role_production_origin"
    ] == [False]
    assert result["removed_ttbar_selected_external_conversion"] == [False]
    assert result["pass_conversion_overlap_removal"] == [expected_pass]


@pytest.mark.parametrize(
    ("sample_name", "expected_role_field", "expected_removed"),
    [
        (
            "UL18_TTGJets",
            "sample_role_is_ttgamma_production",
            True,
        ),
        (
            "UL18_TTGamma_Dilept",
            "sample_role_is_ttgamma_decay",
            True,
        ),
        (
            "TTG-1Jets_PTG-10to100_2022",
            "sample_role_is_ttgamma_inclusive",
            False,
        ),
        (
            "UL18_TTTo2L2Nu",
            "sample_role_is_inclusive_ttbar",
            True,
        ),
    ],
)
def test_mixed_origin_truth_table_and_role_diagnostics(
    sample_name,
    expected_role_field,
    expected_removed,
):
    events, result = _attach_overlap(sample_name, [True], [True])

    assert result[
        "has_mixed_decay_and_production_conversion_photons"
    ] == [True]
    assert result["removed_by_conversion_overlap_removal"] == [
        expected_removed
    ]
    assert result["pass_conversion_overlap_removal"] == [
        not expected_removed
    ]
    assert result[expected_role_field] == [True]
    assert (
        "ttgamma_photon_history_"
        "has_mixed_decay_and_production_conversion_photons"
    ) in events


def test_non_conversion_lepton_is_not_a_candidate():
    result = _classify(
        [[_genpart(22)]],
        [[_lepton(0, gen_part_flav=1)]],
    )

    assert _categories(result) == [[NOT_CONVERSION]]
    assert _event(result, "has_selected_conversion_lepton") == [False]
    assert _event(result, "n_recovered_conversion_photons") == [0]
    assert _event(result, "n_classified_origin_conversion_photons") == [0]


def test_direct_photon_match_and_decay_lepton_classification():
    result = _classify(
        [[_genpart(11), _genpart(22, 0)]],
        [[_lepton(1)]],
    )

    assert _categories(result) == [[DECAY_LEPTON]]
    assert ak.to_list(result["lepton"]["recovered_photon_index"]) == [[1]]
    assert ak.to_list(result["lepton"]["first_copy_photon_index"]) == [[1]]
    assert _event(result, "has_recovered_conversion_photon") == [True]
    assert _event(result, "has_classified_origin_conversion_photon") == [True]
    assert _event(result, "has_decay_origin_conversion_photon") == [True]


def test_recovers_photon_from_immediate_mother():
    result = _classify(
        [[_genpart(1), _genpart(22, 0), _genpart(11, 1)]],
        [[_lepton(2)]],
    )

    assert _categories(result) == [[PRODUCTION_ISR]]
    assert ak.to_list(result["lepton"]["recovered_photon_index"]) == [[1]]


def test_recovers_photon_from_bounded_ancestor_chain():
    result = _classify(
        [[
            _genpart(1),
            _genpart(22, 0),
            _genpart(11, 1),
            _genpart(11, 2),
        ]],
        [[_lepton(3)]],
    )

    assert _categories(result) == [[PRODUCTION_ISR]]
    assert ak.to_list(result["lepton"]["recovered_photon_index"]) == [[1]]


def test_invalid_initial_match_and_no_photon_found_are_distinct():
    result = _classify(
        [[_genpart(11, -1)]],
        [[_lepton(9), _lepton(0)]],
    )

    assert _categories(result) == [[INVALID_MATCH, NO_PHOTON_FOUND]]
    assert _event(result, "has_recovered_conversion_photon") == [False]
    assert _event(result, "has_classified_origin_conversion_photon") == [False]
    assert _event(result, "has_invalid_match_conversion_lepton") == [True]
    assert _event(result, "has_no_photon_found_conversion_lepton") == [True]


def test_first_copy_normalization_walks_same_pdg_photon_mothers():
    result = _classify(
        [[_genpart(1), _genpart(22, 0), _genpart(22, 1)]],
        [[_lepton(2)]],
    )

    assert _categories(result) == [[PRODUCTION_ISR]]
    assert ak.to_list(result["lepton"]["recovered_photon_index"]) == [[2]]
    assert ak.to_list(result["lepton"]["first_copy_photon_index"]) == [[1]]


def test_decay_w_or_b_requires_a_top_ancestor():
    result = _classify(
        [[_genpart(6), _genpart(24, 0), _genpart(22, 1)]],
        [[_lepton(2)]],
    )

    assert _categories(result) == [[DECAY_W_OR_B_WITH_TOP_ANCESTOR]]


def test_decay_top_copy_condition_uses_signed_top_copy():
    result = _classify(
        [[_genpart(-6), _genpart(-6, 0), _genpart(22, 1)]],
        [[_lepton(2)]],
    )

    assert _categories(result) == [[DECAY_TOP_COPY_CONDITION]]


def test_production_isr_and_offshell_top_categories():
    result = _classify(
        [[
            _genpart(1),
            _genpart(22, 0),
            _genpart(21),
            _genpart(6, 2),
            _genpart(22, 3),
        ]],
        [[_lepton(1), _lepton(4)]],
    )

    assert _categories(result) == [[PRODUCTION_ISR, PRODUCTION_OFFSHELL_TOP]]
    assert _event(result, "n_production_origin_conversion_photons") == [2]


def test_gluon_mother_is_offshell_top_production_category():
    result = _classify(
        [[_genpart(21), _genpart(22, 0)]],
        [[_lepton(1)]],
    )

    assert _categories(result) == [[PRODUCTION_OFFSHELL_TOP]]


def test_hadron_ancestor_has_precedence_over_origin_category():
    result = _classify(
        [[_genpart(111), _genpart(22, 0)]],
        [[_lepton(1)]],
    )

    assert _categories(result) == [[HADRON_ANCESTOR]]
    assert _event(result, "has_recovered_conversion_photon") == [True]
    assert _event(result, "has_classified_origin_conversion_photon") == [False]
    assert _event(result, "has_hadron_ancestor_conversion_photon") == [True]
    assert _event(result, "has_decay_origin_conversion_photon") == [False]
    assert _event(result, "has_production_origin_conversion_photon") == [False]


def test_no_mother_and_malformed_cycle_are_ambiguous():
    result = _classify(
        [
            [_genpart(22, -1)],
            [_genpart(22, 1), _genpart(22, 0)],
        ],
        [
            [_lepton(0)],
            [_lepton(0)],
        ],
        max_depth=4,
    )

    assert _categories(result) == [[AMBIGUOUS], [AMBIGUOUS]]
    assert _event(result, "has_recovered_conversion_photon") == [True, True]
    assert _event(result, "has_classified_origin_conversion_photon") == [
        False,
        False,
    ]
    assert _event(result, "has_ambiguous_conversion_photon") == [True, True]


def test_multiple_conversion_leptons_reduce_to_event_counts():
    result = _classify(
        [[
            _genpart(11),
            _genpart(22, 0),
            _genpart(1),
            _genpart(22, 2),
        ]],
        [[_lepton(1), _lepton(3), _lepton(0, gen_part_flav=1)]],
    )

    assert _categories(result) == [
        [DECAY_LEPTON, PRODUCTION_ISR, NOT_CONVERSION]
    ]
    assert _event(result, "n_selected_conversion_leptons") == [2]
    assert _event(result, "n_recovered_conversion_photons") == [2]
    assert _event(result, "n_classified_origin_conversion_photons") == [2]
    assert _event(result, "n_decay_origin_conversion_photons") == [1]
    assert _event(result, "n_production_origin_conversion_photons") == [1]


def test_recovered_and_classified_origin_partitions_include_guard_categories():
    result = _classify(
        [[
            _genpart(11),
            _genpart(22, 0),
            _genpart(1),
            _genpart(22, 2),
            _genpart(111),
            _genpart(22, 4),
            _genpart(22),
            _genpart(11),
        ]],
        [[
            _lepton(1),
            _lepton(3),
            _lepton(5),
            _lepton(6),
            _lepton(7),
            _lepton(99),
            _lepton(1, gen_part_flav=1),
        ]],
    )

    assert _categories(result) == [[
        DECAY_LEPTON,
        PRODUCTION_ISR,
        HADRON_ANCESTOR,
        AMBIGUOUS,
        NO_PHOTON_FOUND,
        INVALID_MATCH,
        NOT_CONVERSION,
    ]]
    assert ak.to_list(
        result["lepton"]["has_recovered_conversion_photon"]
    ) == [[True, True, True, True, False, False, False]]

    event = result["event"]
    assert ak.to_list(event["n_recovered_conversion_photons"]) == [4]
    assert ak.to_list(event["n_classified_origin_conversion_photons"]) == [2]
    assert ak.to_list(event["n_decay_origin_conversion_photons"]) == [1]
    assert ak.to_list(event["n_production_origin_conversion_photons"]) == [1]
    assert ak.to_list(event["n_hadron_ancestor_conversion_photons"]) == [1]
    assert ak.to_list(event["n_ambiguous_conversion_photons"]) == [1]
    assert ak.to_list(event["n_no_photon_found_conversion_leptons"]) == [1]
    assert ak.to_list(event["n_invalid_match_conversion_leptons"]) == [1]

    assert ak.to_list(event["n_recovered_conversion_photons"]) == ak.to_list(
        event["n_classified_origin_conversion_photons"]
        + event["n_hadron_ancestor_conversion_photons"]
        + event["n_ambiguous_conversion_photons"]
    )
    assert ak.to_list(
        event["n_classified_origin_conversion_photons"]
    ) == ak.to_list(
        event["n_decay_origin_conversion_photons"]
        + event["n_production_origin_conversion_photons"]
    )
    assert ak.to_list(
        event["has_classified_origin_conversion_photon"]
    ) == ak.to_list(
        event["has_decay_origin_conversion_photon"]
        | event["has_production_origin_conversion_photon"]
    )


def test_empty_event_is_supported():
    result = _classify(
        [[], [_genpart(1), _genpart(22, 0)]],
        [[], [_lepton(1)]],
    )

    assert _categories(result) == [[], [PRODUCTION_ISR]]
    assert _event(result, "n_selected_conversion_leptons") == [0, 1]


def test_missing_gen_or_lepton_branches_returns_false_diagnostics():
    leptons = ak.Array([[{"pt": 25.0}], []])
    result = classify_selected_conversion_photon_history(None, leptons)

    assert _categories(result) == [[NOT_CONVERSION], []]
    assert _event(result, "diagnostic_missing_branches") == [True, True]
    assert _event(result, "has_selected_conversion_lepton") == [False, False]
    assert _event(result, "n_selected_conversion_leptons") == [0, 0]


def test_attachment_path_handles_data_like_missing_gen_fields():
    events = {}
    leptons = ak.Array([[{"pt": 25.0}], []])

    attached = attach_photon_history_diagnostics(
        events,
        leptons,
        genparts=None,
    )

    assert isinstance(attached, ak.Array)
    assert not isinstance(attached, tuple)
    assert "conversion_photon_history_category" in ak.fields(attached)
    assert (
        "conversion_photon_history_recovered_photon_index"
        in ak.fields(attached)
    )
    assert (
        "conversion_photon_history_has_recovered_conversion_photon"
        in ak.fields(attached)
    )
    stale_index_field = (
        "conversion_photon_history_" + "matched_" + "photon_index"
    )
    assert stale_index_field not in ak.fields(attached)
    stale_lepton_flag = (
        "conversion_photon_history_" + "has_" + "matched_conversion_photon"
    )
    assert (
        stale_lepton_flag not in ak.fields(attached)
    )
    assert ak.to_list(
        events["ttgamma_photon_history_diagnostic_missing_branches"]
    ) == [True, True]
    assert (
        "ttgamma_photon_history_has_classified_origin_conversion_photon"
        in events
    )
    assert "ttgamma_photon_history_has_decay_origin_conversion_photon" in events
    assert (
        "ttgamma_photon_history_has_production_origin_conversion_photon"
        in events
    )
    stale_event_flag = (
        "ttgamma_photon_history_" + "has_" + "matched_conversion_photon"
    )
    stale_event_count = (
        "ttgamma_photon_history_" + "n_" + "matched_conversion_photons"
    )
    assert stale_event_flag not in events
    assert stale_event_count not in events
    assert ak.to_list(
        events["ttgamma_photon_history_has_selected_conversion_lepton"]
    ) == [False, False]


def test_processor_attaches_and_consumes_overlap_removal_diagnostics():
    processor_source = (
        REPO_ROOT / "analysis" / "topeft_run2" / "analysis_processor.py"
    ).read_text()
    selection_source = (
        REPO_ROOT / "topeft" / "modules" / "event_selection.py"
    ).read_text()

    attach_position = processor_source.index(
        "attach_photon_history_diagnostics("
    )
    overlap_position = processor_source.index(
        "attach_conversion_overlap_removal_diagnostics(", attach_position
    )
    event_selection_position = processor_source.index(
        "te_es.add1lMaskAndSFs(", overlap_position
    )
    final_mask_position = processor_source.index(
        '"pass_conversion_overlap_removal"', event_selection_position
    )

    assert attach_position < overlap_position < event_selection_position
    assert event_selection_position < final_mask_position
    assert "ttgamma_sample_role_policy=\"split\"" in processor_source
    assert "self._ttgamma_sample_role_policy = get_ttgamma_sample_role_policy" in (
        processor_source
    )
    assert "sample_role_policy=self._ttgamma_sample_role_policy" in (
        processor_source
    )
    assert "all_cuts_mask" in processor_source[
        final_mask_position - 250 : final_mask_position + 250
    ]
    assert "ttgamma_photon_history_" not in selection_source
    assert "conversion_photon_history_" not in selection_source


def test_ttgamma_sample_role_policy_defaults_are_split_source_guards():
    module_source = (
        REPO_ROOT / "topeft" / "modules" / "ttgamma_photon_history.py"
    ).read_text()
    processor_source = (
        REPO_ROOT / "analysis" / "topeft_run2" / "analysis_processor.py"
    ).read_text()
    run_analysis_source = (
        REPO_ROOT / "analysis" / "topeft_run2" / "run_analysis.py"
    ).read_text()
    full_r3_source = (
        REPO_ROOT / "analysis" / "topeft_run2" / "fullR3_run.sh"
    ).read_text()
    run_ttgamma_source = (
        REPO_ROOT / "analysis" / "topeft_run2" / "run_ttgamma.sh"
    ).read_text()

    assert 'SPLIT_SAMPLE_ROLE_POLICY = "split"' in module_source
    assert (
        "RUN2_NLO_INCLUSIVE_SAMPLE_ROLE_POLICY = "
        '"run2_nlo_inclusive"'
    ) in module_source
    assert "sample_role_policy=SPLIT_SAMPLE_ROLE_POLICY" in module_source
    assert "os.environ" not in module_source
    assert "TOPEFT_TTGAMMA_SAMPLE_ROLE_POLICY" not in module_source
    assert "CL007AT_POLICY_MODE" not in module_source

    assert 'ttgamma_sample_role_policy="split"' in processor_source
    assert "default=SPLIT_SAMPLE_ROLE_POLICY" in run_analysis_source
    assert 'local TTGAMMA_SAMPLE_ROLE_POLICY="split"' in full_r3_source
    assert (
        'RUN_CMD+=(--ttgamma-sample-role-policy '
        '"$TTGAMMA_SAMPLE_ROLE_POLICY")'
    ) in full_r3_source
    assert 'policyMode="split"' in run_ttgamma_source
    assert 'campaignTag="rolepolicy_v2"' in run_ttgamma_source
    assert 'campaignTag="rolepolicy_v2_run2NloInclusive"' in (
        run_ttgamma_source
    )
    assert "CL007AT_POLICY_MODE" not in run_ttgamma_source
    assert "TOPEFT_TTGAMMA_SAMPLE_ROLE_POLICY" not in run_ttgamma_source


def test_stale_matched_field_names_do_not_reappear():
    module_source = (
        REPO_ROOT / "topeft" / "modules" / "ttgamma_photon_history.py"
    ).read_text()
    processor_source = (
        REPO_ROOT / "analysis" / "topeft_run2" / "analysis_processor.py"
    ).read_text()

    stale_names = (
        "has_matched_conversion_photon",
        "n_matched_conversion_photons",
        "matched_photon_index",
    )
    for stale_name in stale_names:
        assert stale_name not in module_source
        assert stale_name not in processor_source
