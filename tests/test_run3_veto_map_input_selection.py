import awkward as ak
from pathlib import Path

from analysis.topeft_run2.analysis_processor import get_veto_map_input_jets


def _processor_source():
    return (
        Path(__file__).parents[1]
        / "analysis"
        / "topeft_run2"
        / "analysis_processor.py"
    ).read_text()


def test_get_veto_map_input_jets_applies_run3_minimal_selection():
    jets = ak.Array(
        [
            [
                {
                    "pt": 20.0,
                    "eta": 0.2,
                    "jetId": 2,
                    "chHEF": 0.5,
                    "neHEF": 0.1,
                    "neEmEF": 0.1,
                    "muEF": 0.1,
                    "chEmEF": 0.1,
                },
                {
                    "pt": 15.0,
                    "eta": 0.2,
                    "jetId": 2,
                    "chHEF": 0.5,
                    "neHEF": 0.1,
                    "neEmEF": 0.1,
                    "muEF": 0.1,
                    "chEmEF": 0.1,
                },
                {
                    "pt": 30.0,
                    "eta": 0.2,
                    "jetId": 0,
                    "chHEF": 0.5,
                    "neHEF": 0.1,
                    "neEmEF": 0.1,
                    "muEF": 0.1,
                    "chEmEF": 0.1,
                },
                {
                    "pt": 40.0,
                    "eta": 0.2,
                    "jetId": 2,
                    "chHEF": 0.5,
                    "neHEF": 0.1,
                    "neEmEF": 0.35,
                    "muEF": 0.1,
                    "chEmEF": 0.6,
                },
                {
                    "pt": 50.0,
                    "eta": 0.2,
                    "jetId": 2,
                    "chHEF": 0.5,
                    "neHEF": 0.1,
                    "neEmEF": 0.02,
                    "muEF": 0.1,
                    "chEmEF": 0.85,
                },
                {
                    "pt": 80.0,
                    "eta": 3.2,
                    "jetId": 2,
                    "chHEF": 0.5,
                    "neHEF": 0.1,
                    "neEmEF": 0.1,
                    "muEF": 0.9,
                    "chEmEF": 0.0,
                },
            ]
        ]
    )

    selected = get_veto_map_input_jets(jets, "2022", True)

    assert ak.to_list(selected.pt) == [[20.0, 80.0]]


def test_get_veto_map_input_jets_uses_received_pt_field():
    jets = ak.Array(
        [
            [
                {
                    "pt": 16.0,
                    "pt_raw": 10.0,
                    "eta": 0.2,
                    "jetId": 2,
                    "chHEF": 0.5,
                    "neHEF": 0.1,
                    "neEmEF": 0.1,
                    "muEF": 0.1,
                    "chEmEF": 0.1,
                },
                {
                    "pt": 14.0,
                    "pt_raw": 30.0,
                    "eta": 0.2,
                    "jetId": 2,
                    "chHEF": 0.5,
                    "neHEF": 0.1,
                    "neEmEF": 0.1,
                    "muEF": 0.1,
                    "chEmEF": 0.1,
                },
            ]
        ]
    )

    selected = get_veto_map_input_jets(jets, "2022", True)

    assert ak.to_list(selected.pt) == [[16.0]]
    assert ak.to_list(selected.pt_raw) == [[10.0]]


def test_get_veto_map_input_jets_keeps_jme_selection_looser_than_analysis_jets():
    jets = ak.Array(
        [
            [
                {
                    "pt": 16.0,
                    "eta": 0.2,
                    "jetId": 2,
                    "chHEF": 0.5,
                    "neHEF": 0.1,
                    "neEmEF": 0.1,
                    "muEF": 0.1,
                    "chEmEF": 0.1,
                }
            ]
        ]
    )

    selected = get_veto_map_input_jets(jets, "2022", True)

    assert ak.to_list(selected.pt) == [[16.0]]


def test_get_veto_map_input_jets_preserves_non_run3_inputs():
    jets = ak.Array([[{"pt": 10.0}, {"pt": 20.0}]])

    selected = get_veto_map_input_jets(jets, "2018", False)

    assert ak.to_list(selected.pt) == [[10.0, 20.0]]


def test_processor_applies_run3_veto_maps_after_jet_corrections_and_systematics():
    source = _processor_source()

    cleaning = source.index("cleanedJets = jets[~ak.any(tmp.slot0 == tmp.slot1, axis=-1)]")
    raw_attachment = source.index('cleanedJets["pt_raw"] =')
    corrections = source.index("cleanedJets = ApplyJetCorrections(")
    systematics = source.index("cleanedJets = apply_maintained_jet_systematic(")
    veto_inputs = source.index(
        "veto_map_input_jets = get_veto_map_input_jets(cleanedJets, year, is_run3)"
    )
    veto_eval = source.index(
        "veto_map_array = ApplyJetVetoMaps(veto_map_input_jets, year) if is_run3 else ak.zeros_like(met.pt)"
    )
    analysis_jet_selection = source.index('cleanedJets["isGood"]')

    assert (
        cleaning
        < raw_attachment
        < corrections
        < systematics
        < veto_inputs
        < veto_eval
        < analysis_jet_selection
    )


def test_processor_keeps_run2_veto_maps_disabled():
    source = _processor_source()

    assert (
        "veto_map_array = ApplyJetVetoMaps(veto_map_input_jets, year) if is_run3 else ak.zeros_like(met.pt)"
        in source
    )
