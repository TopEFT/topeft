import awkward as ak

from analysis.topeft_run2.analysis_processor import get_veto_map_input_jets


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


def test_get_veto_map_input_jets_preserves_non_run3_inputs():
    jets = ak.Array([[{"pt": 10.0}, {"pt": 20.0}]])

    selected = get_veto_map_input_jets(jets, "2018", False)

    assert ak.to_list(selected.pt) == [[10.0, 20.0]]
