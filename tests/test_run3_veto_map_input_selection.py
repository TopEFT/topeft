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

    corrected_view_start = source.index("def build_corrected_jet_view(")
    corrected_view_end = source.index(
        "\ndef build_analysis_and_hem_jet_views(", corrected_view_start
    )
    corrected_view = source[corrected_view_start:corrected_view_end]
    raw_attachment = corrected_view.index('jets["pt_raw"] =')
    corrections = corrected_view.index("correction_factory.build(")
    systematics = corrected_view.index("apply_maintained_jet_systematic(")

    analysis_view_start = source.index("def build_analysis_and_hem_jet_views(")
    analysis_view_end = source.index("\ndef is_in_hem2018_region", analysis_view_start)
    analysis_view = source[analysis_view_start:analysis_view_end]
    cleaning = analysis_view.index("analysis_raw_jets = get_analysis_cleaned_jets(")
    analysis_correction = analysis_view.index(
        "analysis_corrected_jets, jet_pt_name = build_corrected_jet_view("
    )
    non_2018_assignment = analysis_view.index(
        "hem_corrected_jets = analysis_corrected_jets"
    )

    jet_section = source.index("#################### Jets ####################")
    view_build = source.index("build_analysis_and_hem_jet_views(", jet_section)
    veto_inputs = source.index(
        "veto_map_input_jets = get_veto_map_input_jets(cleanedJets, year, is_run3)"
    )
    veto_eval = source.index(
        "veto_map_array = ApplyJetVetoMaps(veto_map_input_jets, year) if is_run3 else ak.zeros_like(met.pt)"
    )
    analysis_jet_selection = source.index('cleanedJets["isGood"]')

    assert raw_attachment < corrections < systematics
    assert cleaning < analysis_correction < non_2018_assignment
    assert view_build < veto_inputs < veto_eval < analysis_jet_selection


def test_processor_keeps_run2_out_of_run3_event_veto_route():
    source = _processor_source()

    assert (
        "veto_map_array = ApplyJetVetoMaps(veto_map_input_jets, year) if is_run3 else ak.zeros_like(met.pt)"
        in source
    )
    assert "if is_run2_jvm_year(year):" in source
    assert "cleanedJets = apply_run2_jvm_to_analysis_jets(" in source
