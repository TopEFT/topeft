from pathlib import Path
import awkward as ak
import numpy as np
import pytest

from analysis.topeft_run2.analysis_processor import (
    apply_run2_jvm_to_analysis_jets,
    get_run2_jvm_keep_mask,
    get_run2_jvm_qualifying_jet_mask,
    is_run2_jvm_year,
)
from topeft.modules import corrections


def _processor_source():
    return (
        Path(__file__).parents[1]
        / "analysis"
        / "topeft_run2"
        / "analysis_processor.py"
    ).read_text()


def _jet(
    jet_id_marker,
    *,
    pt=30.0,
    eta=0.0,
    phi=0.0,
    pu_id=4,
    jet_id_bits=2,
    ch_em_ef=0.1,
    ne_em_ef=0.1,
):
    return {
        "jet_id_marker": jet_id_marker,
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "jetId": jet_id_bits,
        "puId": pu_id,
        "chEmEF": ch_em_ef,
        "neEmEF": ne_em_ef,
    }


def _pf_muon(*, eta=2.0, phi=2.0):
    return {"eta": eta, "phi": phi}


@pytest.mark.parametrize(
    ("year", "expected_payload_dir", "expected_key"),
    [
        ("2016APV", "2016preVFP_UL", "Summer19UL16_V1"),
        ("2016", "2016postVFP_UL", "Summer19UL16_V1"),
        ("2017", "2017_UL", "Summer19UL17_V1"),
        ("2018", "2018_UL", "Summer19UL18_V1"),
    ],
)
def test_run2_jvm_dispatches_exact_payload_and_category(
    monkeypatch, year, expected_payload_dir, expected_key
):
    observed = {"paths": [], "keys": [], "calls": []}

    class FakeCorrection:
        def evaluate(self, *args):
            observed["calls"].append(args)
            return np.zeros_like(args[1])

    class FakeCorrectionSet:
        def __getitem__(self, key):
            observed["keys"].append(key)
            return FakeCorrection()

    class FakeFactory:
        @staticmethod
        def from_file(path):
            observed["paths"].append(path)
            return FakeCorrectionSet()

    monkeypatch.setattr(corrections, "topcoffea_path", lambda path: path)
    monkeypatch.setattr(
        corrections.correctionlib,
        "CorrectionSet",
        FakeFactory,
    )
    jets = ak.Array(
        [[{"eta": 0.1, "phi": 0.2}], [], [{"eta": -0.3, "phi": -0.4}]]
    )

    scores = corrections.get_run2_jet_veto_map_scores(jets, year)

    assert observed["paths"] == [
        f"data/POG/JME/{expected_payload_dir}/jetvetomaps.json.gz"
    ]
    assert observed["keys"] == [expected_key]
    assert observed["calls"][0][0] == "jetvetomap_all"
    assert ak.to_list(scores) == [[0.0], [], [0.0]]


@pytest.mark.parametrize("year", ["2022", "2022EE", "2023", "2023BPix"])
def test_non_run2_years_do_not_enter_per_jet_jvm(year):
    assert not is_run2_jvm_year(year)
    with pytest.raises(ValueError, match="Run-2 jet veto maps are not defined"):
        corrections.get_run2_jet_veto_map_scores(
            ak.Array([[{"eta": 0.0, "phi": 0.0}]]), year
        )


def test_run2_jvm_qualifying_boundaries():
    jets = ak.Array(
        [
            [_jet(0, pt=15.0)],
            [_jet(1, pt=15.01)],
            [_jet(2, jet_id_bits=0)],
            [_jet(3, ch_em_ef=0.4, ne_em_ef=0.5)],
            [_jet(4, ch_em_ef=0.4, ne_em_ef=0.499)],
            [_jet(5, pt=60.0, pu_id=0)],
            [_jet(6)],
            [_jet(7)],
        ]
    )
    pf_muons = ak.Array(
        [
            [_pf_muon()],
            [_pf_muon()],
            [_pf_muon()],
            [_pf_muon()],
            [_pf_muon()],
            [_pf_muon()],
            [_pf_muon(eta=0.1, phi=0.0)],
            [_pf_muon(eta=0.2, phi=0.0)],
        ]
    )

    qualifying = get_run2_jvm_qualifying_jet_mask(jets, pf_muons, "2018")

    assert ak.to_list(ak.flatten(qualifying)) == [
        False,
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]


@pytest.mark.parametrize(
    ("year", "passing_bit", "wrong_bit"),
    [
        ("2016APV", 1, 4),
        ("2016", 1, 4),
        ("2017", 4, 1),
        ("2018", 4, 1),
    ],
)
def test_run2_jvm_uses_era_specific_loose_pu_id_bit(
    year, passing_bit, wrong_bit
):
    jets = ak.Array(
        [[_jet(0, pu_id=passing_bit), _jet(1, pu_id=wrong_bit)]]
    )
    pf_muons = ak.Array([[_pf_muon()]])

    qualifying = get_run2_jvm_qualifying_jet_mask(jets, pf_muons, year)

    assert ak.to_list(qualifying) == [[True, False]]


def test_run2_jvm_removes_only_qualifying_map_vetoed_jets():
    jets = ak.Array(
        [
            [
                _jet(0),
                _jet(1),
                _jet(2, pt=15.0),
            ],
            [_jet(3)],
        ]
    )
    pf_muons = ak.Array([[_pf_muon()], [_pf_muon()]])

    def map_evaluator(received_jets, year):
        assert received_jets is jets
        assert year == "2018"
        return ak.Array([[1.0, 0.0, 1.0], [0.0]])

    keep_mask = get_run2_jvm_keep_mask(
        jets, pf_muons, "2018", map_evaluator
    )
    filtered = apply_run2_jvm_to_analysis_jets(
        jets, pf_muons, "2018", map_evaluator
    )

    assert ak.to_list(keep_mask) == [[False, True, True], [True]]
    assert ak.to_list(filtered.jet_id_marker) == [[1, 2], [3]]
    assert len(filtered) == len(jets)


def test_run2_jvm_data_and_mc_share_one_per_jet_contract():
    jets = ak.Array([[_jet(0), _jet(1, pt=15.0)]])
    pf_muons = ak.Array([[_pf_muon()]])
    map_evaluator = lambda received_jets, year: ak.Array([[1.0, 1.0]])

    data_keep_mask = get_run2_jvm_keep_mask(
        jets, pf_muons, "2017", map_evaluator
    )
    mc_keep_mask = get_run2_jvm_keep_mask(
        jets, pf_muons, "2017", map_evaluator
    )

    assert "is_data" not in get_run2_jvm_keep_mask.__code__.co_varnames
    assert ak.to_list(data_keep_mask) == ak.to_list(mc_keep_mask) == [
        [False, True]
    ]


def test_run2_jvm_recomputes_on_received_active_jet_view():
    nominal_jets = ak.Array([[_jet(0, pt=15.0)]])
    varied_jets = ak.with_field(nominal_jets, [[15.01]], "pt")
    pf_muons = ak.Array([[_pf_muon()]])
    map_evaluator = lambda received_jets, year: ak.ones_like(received_jets.pt)

    nominal = apply_run2_jvm_to_analysis_jets(
        nominal_jets, pf_muons, "2018", map_evaluator
    )
    varied = apply_run2_jvm_to_analysis_jets(
        varied_jets, pf_muons, "2018", map_evaluator
    )

    assert ak.to_list(nominal.jet_id_marker) == [[0]]
    assert ak.to_list(varied.jet_id_marker) == [[]]


def test_run2_jvm_keeps_event_and_changes_downstream_jet_count():
    jets = ak.Array(
        [[_jet(0, pu_id=1), _jet(1, pu_id=1)], [_jet(2, pu_id=1)]]
    )
    pf_muons = ak.Array([[_pf_muon()], [_pf_muon()]])
    map_evaluator = lambda received_jets, year: ak.Array(
        [[1.0, 0.0], [0.0]]
    )

    filtered = apply_run2_jvm_to_analysis_jets(
        jets, pf_muons, "2016APV", map_evaluator
    )

    assert len(filtered) == 2
    assert ak.to_list(ak.num(jets)) == [2, 1]
    assert ak.to_list(ak.num(filtered)) == [1, 1]
    assert ak.to_list(filtered.jet_id_marker) == [[1], [2]]


def test_processor_routes_only_analysis_jets_through_run2_jvm():
    source = _processor_source()
    jet_section = source.index("#################### Jets ####################")
    view_build = source.index("build_analysis_and_hem_jet_views(", jet_section)
    hem_call = source.index("hem2018_mask = get_hem2018_event_mask(", view_build)
    hem_full_view = source.index("                hem_corrected_jets,", hem_call)
    met_selection = source.index("if use_type1_met(year):", hem_call)
    run2_filter = source.index("if is_run2_jvm_year(year):", met_selection)
    analysis_selection = source.index('cleanedJets["isGood"]', run2_filter)
    good_jets = source.index("goodJets = cleanedJets[cleanedJets.isGood]", run2_filter)
    jet_count = source.index("njets = ak.num(goodJets)", good_jets)
    btag_count = source.index("isBtagJetsLoose = (goodJets[btagAlgo] > btagwpl)", jet_count)

    assert view_build < hem_call < hem_full_view < met_selection < run2_filter
    assert run2_filter < analysis_selection < good_jets < jet_count < btag_count
    assert "apply_run2_jvm_to_analysis_jets(\n                    cleanedJets," in source
    assert "apply_run2_jvm_to_analysis_jets(\n                    hem_corrected_jets," not in source
    assert "if is_run2_jvm_year(year):" in source
    assert "if is_run3:" in source


def test_processor_preserves_run3_event_veto_and_met_ownership():
    source = _processor_source()

    assert (
        "veto_map_array = ApplyJetVetoMaps(veto_map_input_jets, year) "
        "if is_run3 else ak.zeros_like(met.pt)"
    ) in source
    assert 'selections.add("jet_veto", veto_map_mask)' in source
    assert "met = ApplyMETSystematics(type1_met, syst_var)" in source
    assert "should_fill_jvm_eta_phi_diagnostic(is_run3, syst_var, wgt_fluct)" in source
