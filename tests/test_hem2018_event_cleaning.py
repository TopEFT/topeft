from pathlib import Path

import awkward as ak
import numpy as np
from coffea.nanoevents.methods import candidate

import analysis.topeft_run2.analysis_processor as analysis_processor
from analysis.topeft_run2.analysis_processor import (
    build_analysis_and_hem_jet_views,
    get_analysis_cleaned_jets,
    get_hem2018_affected_mc_mask,
    get_hem2018_event_mask,
    get_hem2018_qualifying_jet_mask,
    hem2018_affected_lumi_fraction,
    is_in_hem2018_region,
)


def _processor_source():
    return (
        Path(__file__).parents[1]
        / "analysis"
        / "topeft_run2"
        / "analysis_processor.py"
    ).read_text()


def _candidate_array(events):
    return ak.with_name(
        ak.Array(events),
        "PtEtaPhiMCandidate",
        behavior=candidate.behavior,
    )


def _jet(
    *,
    pt=30.0,
    eta=-2.0,
    phi=-1.0,
    jet_id=6,
    pu_id=4,
    ch_em_ef=0.1,
    ne_em_ef=0.1,
    btag=0.0,
    raw_index=0,
):
    return {
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "mass": 0.0,
        "jetId": jet_id,
        "puId": pu_id,
        "chEmEF": ch_em_ef,
        "neEmEF": ne_em_ef,
        "btagDeepFlavB": btag,
        "rawFactor": 0.0,
        "area": 0.5,
        "raw_index": raw_index,
    }


def _pf_muon(*, eta=2.0, phi=2.0):
    return {
        "pt": 10.0,
        "eta": eta,
        "phi": phi,
        "mass": 0.105,
        "isPFcand": True,
    }


def _affected_data_mask(jets, pf_muons):
    event_count = len(jets)
    return get_hem2018_event_mask(
        jets,
        pf_muons,
        np.arange(event_count, dtype=np.uint64),
        np.full(event_count, 319077),
        "2018",
        True,
    )


def test_hem2018_data_run_boundary_and_event_veto_semantics():
    jets = _candidate_array(
        [
            [_jet()],
            [_jet()],
            [_jet()],
            [_jet(pt=15.0)],
        ]
    )
    pf_muons = _candidate_array([[_pf_muon()] for _ in range(4)])
    runs = np.array([319076, 319077, 325175, 325175])
    event_numbers = np.array([1, 2, 3, 4], dtype=np.uint64)

    event_mask = get_hem2018_event_mask(
        jets, pf_muons, event_numbers, runs, "2018", True
    )

    assert ak.to_list(event_mask) == [True, False, False, True]


def test_hem2018_non_2018_data_is_unchanged():
    jets = _candidate_array([[_jet()]])
    pf_muons = _candidate_array([[_pf_muon()]])

    event_mask = get_hem2018_event_mask(
        jets,
        pf_muons,
        np.array([1], dtype=np.uint64),
        np.array([325175]),
        "2017",
        True,
    )

    assert ak.to_list(event_mask) == [True]


def test_hem2018_geometry_uses_strict_boundaries():
    jets = ak.Array(
        [
            [
                {"eta": -2.0, "phi": -1.0},
                {"eta": -3.0, "phi": -1.0},
                {"eta": -1.3, "phi": -1.0},
                {"eta": -2.0, "phi": -1.57},
                {"eta": -2.0, "phi": -0.87},
                {"eta": -3.000001, "phi": -1.0},
                {"eta": -1.299999, "phi": -1.0},
                {"eta": -2.0, "phi": -1.570001},
                {"eta": -2.0, "phi": -0.869999},
            ]
        ]
    )

    assert ak.to_list(is_in_hem2018_region(jets)) == [
        [True, False, False, False, False, False, False, False, False]
    ]


def test_hem2018_qualifying_jet_contract():
    jets = _candidate_array(
        [
            [
                _jet(pt=15.0, eta=0.0),
                _jet(pt=15.01, eta=1.0, ch_em_ef=0.7, ne_em_ef=0.3),
                _jet(pt=20.0, eta=2.0, jet_id=2, ch_em_ef=0.4, ne_em_ef=0.4),
                _jet(pt=20.0, eta=3.0, jet_id=2, ch_em_ef=0.5, ne_em_ef=0.4),
                _jet(pt=20.0, eta=4.0, jet_id=2, ch_em_ef=0.6, ne_em_ef=0.4),
                _jet(pt=20.0, eta=5.0, jet_id=2, ch_em_ef=0.4, ne_em_ef=0.4),
                _jet(pt=49.0, eta=6.0, pu_id=0),
                _jet(pt=49.0, eta=7.0, pu_id=4),
                _jet(pt=50.0, eta=8.0, pu_id=0),
                _jet(pt=60.0, eta=9.0, jet_id=0, pu_id=4),
            ]
        ]
    )
    pf_muons = _candidate_array([[_pf_muon(eta=5.1, phi=-1.0)]])

    qualifying = get_hem2018_qualifying_jet_mask(jets, pf_muons)

    assert ak.to_list(qualifying) == [
        [False, True, True, False, False, False, False, True, True, False]
    ]


def test_hem2018_uses_the_received_active_jet_pt():
    jets = _candidate_array([[_jet(pt=16.0), _jet(pt=14.0)]])
    jets["pt_raw"] = [[10.0, 30.0]]
    pf_muons = _candidate_array([[_pf_muon()]])

    qualifying = get_hem2018_qualifying_jet_mask(jets, pf_muons)

    assert ak.to_list(qualifying) == [[True, False]]


def test_hem2018_mc_assignment_is_identity_stable():
    event_numbers = np.array(
        [0, 1, 2, 17, 101, 2**32 + 9, 2**53 + 11, 2**63 + 5],
        dtype=np.uint64,
    )
    reference = get_hem2018_affected_mc_mask(event_numbers)

    assert np.array_equal(reference, get_hem2018_affected_mc_mask(event_numbers))

    order = np.array([5, 0, 7, 2, 1, 6, 4, 3])
    assert np.array_equal(
        reference[order], get_hem2018_affected_mc_mask(event_numbers[order])
    )

    split_recombined = np.concatenate(
        [
            get_hem2018_affected_mc_mask(event_numbers[:3]),
            get_hem2018_affected_mc_mask(event_numbers[3:6]),
            get_hem2018_affected_mc_mask(event_numbers[6:]),
        ]
    )
    assert np.array_equal(reference, split_recombined)

    worker_recombined = np.empty_like(reference)
    for worker_id in range(3):
        worker_indices = np.arange(len(event_numbers))[::3] + worker_id
        worker_indices = worker_indices[worker_indices < len(event_numbers)]
        worker_recombined[worker_indices] = get_hem2018_affected_mc_mask(
            event_numbers[worker_indices]
        )
    assert np.array_equal(reference, worker_recombined)


def test_hem2018_mc_veto_applies_only_to_the_assigned_period():
    event_numbers = np.arange(20, dtype=np.uint64)
    jets = _candidate_array([[_jet()] for _ in event_numbers])
    pf_muons = _candidate_array([[_pf_muon()] for _ in event_numbers])
    affected = get_hem2018_affected_mc_mask(event_numbers)

    event_mask = get_hem2018_event_mask(
        jets,
        pf_muons,
        event_numbers,
        np.zeros(len(event_numbers), dtype=np.uint32),
        "2018",
        False,
    )

    assert np.any(affected)
    assert np.any(~affected)
    assert np.array_equal(np.asarray(event_mask), ~affected)


def test_hem2018_mc_assignment_fraction_and_sequential_distribution():
    # For 200k independent 64-bit hash outputs, 0.005 is about 4.7 binomial
    # standard deviations at the frozen target. Per-1000-event windows must also
    # stay within 0.08 so a long contiguous modulo block cannot pass unnoticed.
    event_numbers = np.arange(200_000, dtype=np.uint64)
    affected = get_hem2018_affected_mc_mask(event_numbers)
    observed_fraction = np.mean(affected)
    window_fractions = np.mean(affected.reshape(-1, 1000), axis=1)

    assert abs(observed_fraction - hem2018_affected_lumi_fraction) < 0.005
    assert np.max(abs(window_fractions - hem2018_affected_lumi_fraction)) < 0.08


def test_fo_lepton_cleaning_cannot_hide_a_hem_jet():
    full_jets = _candidate_array([[_jet()]])
    pf_muons = _candidate_array([[_pf_muon()]])
    fo_leptons = ak.Array([[{"jetIdx": 0}]])

    analysis_jets = get_analysis_cleaned_jets(full_jets, fo_leptons)

    assert ak.to_list(ak.num(analysis_jets)) == [0]
    assert ak.to_list(_affected_data_mask(full_jets, pf_muons)) == [False]
    assert ak.to_list(_affected_data_mask(analysis_jets, pf_muons)) == [True]


def test_tau_cleaning_cannot_hide_a_hem_jet():
    full_jets = _candidate_array([[_jet()]])
    pf_muons = _candidate_array([[_pf_muon()]])
    unmatched_leptons = ak.Array([[{"jetIdx": -1}]])
    cleaning_taus = _candidate_array(
        [[{"pt": 25.0, "eta": -2.1, "phi": -1.0, "mass": 1.777}]]
    )

    analysis_jets = get_analysis_cleaned_jets(
        full_jets, unmatched_leptons, cleaning_taus
    )

    assert ak.to_list(ak.num(analysis_jets)) == [0]
    assert ak.to_list(_affected_data_mask(full_jets, pf_muons)) == [False]
    assert ak.to_list(_affected_data_mask(analysis_jets, pf_muons)) == [True]


def test_analysis_jet_cleaning_still_owns_downstream_jet_membership():
    full_jets = _candidate_array(
        [[_jet(btag=0.9), _jet(eta=0.0, phi=0.0, btag=0.2)]]
    )
    fo_leptons = ak.Array([[{"jetIdx": 0}]])

    analysis_jets = get_analysis_cleaned_jets(full_jets, fo_leptons)

    assert ak.to_list(ak.num(analysis_jets)) == [1]
    assert ak.to_list(analysis_jets.btagDeepFlavB) == [[0.2]]


def test_two_correction_views_preserve_analysis_order_and_full_hem_owner(
    monkeypatch,
):
    full_jets = _candidate_array(
        [[
            _jet(pt=30.0, raw_index=0),
            _jet(pt=40.0, eta=0.0, phi=0.0, raw_index=1),
            _jet(pt=50.0, eta=2.0, phi=2.0, raw_index=2),
        ]]
    )
    fo_leptons = ak.Array([[{"jetIdx": 0}]])
    cleaning_taus = _candidate_array(
        [[{"pt": 25.0, "eta": 0.0, "phi": 0.0, "mass": 1.777}]]
    )
    analysis_cache = {"owner": "analysis"}
    build_calls = []
    systematic_calls = []

    class MembershipDependentFactory:
        def build(self, jets, lazy_cache):
            build_calls.append(
                {
                    "raw_indices": ak.to_list(jets.raw_index),
                    "cache": lazy_cache,
                }
            )
            position = ak.local_index(jets.pt, axis=1)
            return ak.with_field(jets, jets.pt + 100.0 * position, "pt")

    factory = MembershipDependentFactory()
    monkeypatch.setattr(
        analysis_processor,
        "ApplyJetCorrections",
        lambda *args, **kwargs: factory,
    )

    def apply_test_systematic(year, jets, syst_var, jet_systematics):
        systematic_calls.append(
            (year, syst_var, tuple(jet_systematics), ak.to_list(jets.raw_index))
        )
        return ak.with_field(jets, jets.pt + 7.0, "pt")

    monkeypatch.setattr(
        analysis_processor,
        "apply_maintained_jet_systematic",
        apply_test_systematic,
    )

    analysis_jets, hem_jets, jet_pt_name = build_analysis_and_hem_jet_views(
        full_jets,
        fo_leptons,
        cleaning_taus,
        np.array([20.0]),
        year="2018",
        is_data=True,
        run_era="A",
        run=np.array([319077]),
        suppress_forward_eta_stochastic_jer=False,
        syst_var="JER_2018Up",
        jet_systematics=["JER_2018Up"],
        analysis_lazy_cache=analysis_cache,
    )

    assert jet_pt_name == "pt"
    assert [call["raw_indices"] for call in build_calls] == [[[2]], [[0, 1, 2]]]
    assert build_calls[0]["cache"] is analysis_cache
    assert build_calls[1]["cache"] is not analysis_cache
    assert build_calls[1]["cache"] == {}
    assert systematic_calls == [
        ("2018", "JER_2018Up", ("JER_2018Up",), [[2]]),
        ("2018", "JER_2018Up", ("JER_2018Up",), [[0, 1, 2]]),
    ]
    assert ak.to_list(analysis_jets.raw_index) == [[2]]
    assert ak.to_list(analysis_jets.pt) == [[57.0]]
    assert ak.to_list(hem_jets.raw_index) == [[0, 1, 2]]
    assert ak.to_list(hem_jets.pt) == [[37.0, 147.0, 257.0]]

    pf_muons = _candidate_array([[_pf_muon()]])
    assert ak.to_list(_affected_data_mask(hem_jets, pf_muons)) == [False]
    assert ak.to_list(_affected_data_mask(analysis_jets, pf_muons)) == [True]

    build_calls.clear()
    systematic_calls.clear()
    non_2018_cache = {"owner": "non_2018_analysis"}
    non_2018_analysis_jets, non_2018_hem_jets, _ = (
        build_analysis_and_hem_jet_views(
            full_jets,
            fo_leptons,
            cleaning_taus,
            np.array([20.0]),
            year="2022",
            is_data=True,
            run_era="C",
            run=np.array([355862]),
            suppress_forward_eta_stochastic_jer=False,
            syst_var="JER_2022Up",
            jet_systematics=["JER_2022Up"],
            analysis_lazy_cache=non_2018_cache,
        )
    )

    assert [call["raw_indices"] for call in build_calls] == [[[2]]]
    assert build_calls[0]["cache"] is non_2018_cache
    assert systematic_calls == [
        ("2022", "JER_2022Up", ("JER_2022Up",), [[2]])
    ]
    assert non_2018_hem_jets is non_2018_analysis_jets


def test_processor_routes_separate_active_analysis_and_hem_jet_views():
    source = _processor_source()

    jet_section = source.index("#################### Jets ####################")
    view_build = source.index(
        "build_analysis_and_hem_jet_views(", jet_section
    )
    hem_call = source.index("hem2018_mask = get_hem2018_event_mask(")
    hem_input = source.index("                hem_corrected_jets,", hem_call)
    veto_map = source.index("veto_map_input_jets = get_veto_map_input_jets(")
    analysis_jet_selection = source.index('cleanedJets["isGood"]')

    assert view_build < hem_call < hem_input < veto_map < analysis_jet_selection
    assert "cleaning_taus if self.enable_tau_blocks else None" in source
    assert "analysis_raw_jets = get_analysis_cleaned_jets(" in source
    assert "analysis_corrected_jets, jet_pt_name = build_corrected_jet_view(" in source
    assert "hem_corrected_jets, _ = build_corrected_jet_view(" in source
    assert "hem_corrected_jets = analysis_corrected_jets" in source
    assert 'selections.add("hem2018", hem2018_mask)' in source
    assert 'cuts_lst.append("hem2018")' in source


def test_downstream_counts_and_btags_remain_analysis_cleaned():
    source = _processor_source()

    assert "goodJets = cleanedJets[cleanedJets.isGood]" in source
    assert "fwdJets  = cleanedJets[cleanedJets.isFwd]" in source
    assert "njets = ak.num(goodJets)" in source
    assert "isBtagJetsLoose = (goodJets[btagAlgo] > btagwpl)" in source
    assert "isBtagJetsMedium = (goodJets[btagAlgo] > btagwpm)" in source


def test_hem2018_mask_propagates_to_standard_and_companion_fills():
    source = _processor_source()

    cuts_start = source.index("cuts_lst = [appl,lep_chan]")
    hem_cut = source.index('cuts_lst.append("hem2018")', cuts_start)
    final_selection = source.index("all_cuts_mask = selections.all(", hem_cut)
    standard_weights = source.index(
        "weights_flat = weight[all_cuts_mask]", final_selection
    )
    standard_fill = source.index(
        "hout[nominal_histogram_key].fill(**axes_fill_info_dict)",
        standard_weights,
    )
    companion_fill = source.index(
        'hout[dense_axis_name+"_sumw2"].fill(**sumw2_fill_info)',
        standard_fill,
    )

    assert hem_cut < final_selection < standard_weights < standard_fill
    assert standard_fill < companion_fill


def test_hem2018_remains_separate_from_jvm_weights_and_met_policy():
    source = _processor_source()
    helper_start = source.index("def get_hem2018_event_mask(")
    helper_end = source.index("\ndef resolve_category_dict_names", helper_start)
    hem_helpers = source[helper_start:helper_end]

    assert "ApplyJetVetoMaps(veto_map_input_jets, year) if is_run3" in source
    assert 'selections.add("jet_veto", veto_map_mask)' in source
    assert "jetvetomap_hem1516" not in source
    assert "weight" not in hem_helpers.lower()
    assert "nuisance" not in hem_helpers.lower()
    assert "scale factor" not in hem_helpers.lower()
    assert "met = ApplyMETSystematics(type1_met, syst_var)" in source
