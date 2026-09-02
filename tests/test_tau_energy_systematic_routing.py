import pytest

from analysis.topeft_run2 import analysis_processor
from topeft.modules.corrections import (
    ApplyJetSystematics,
    get_supported_jet_systematics,
    get_supported_tau_energy_systematics,
)


@pytest.mark.parametrize("year", ["2018", "2022"])
def test_all_maintained_tau_energy_families_bypass_jet_systematics(
    monkeypatch, year
):
    calls = []
    cleaned_jets = object()
    tau_systematics = get_supported_tau_energy_systematics(year, isData=False)
    jet_systematics = get_supported_jet_systematics(year, isData=False, era=None)

    def record_jet_systematic(*args):
        calls.append(args)
        return object()

    monkeypatch.setattr(
        analysis_processor, "ApplyJetSystematics", record_jet_systematic
    )

    assert {
        label
        for label in ("genTau", "genElectron", "genMuon")
        if any(label in systematic for systematic in tau_systematics)
    } == {"genTau", "genElectron", "genMuon"}
    for systematic in tau_systematics:
        selected_jets = analysis_processor.apply_maintained_jet_systematic(
            year, cleaned_jets, systematic, jet_systematics
        )
        assert selected_jets is cleaned_jets

    assert calls == []


@pytest.mark.parametrize("year", ["2018", "2022"])
def test_maintained_jet_systematics_retain_the_jet_helper_path(monkeypatch, year):
    calls = []
    cleaned_jets = object()
    varied_jets = object()
    jet_systematics = get_supported_jet_systematics(year, isData=False, era=None)
    representative_systematics = (
        next(
            systematic
            for systematic in jet_systematics
            if systematic.startswith("JER_")
        ),
        next(
            systematic
            for systematic in jet_systematics
            if systematic.startswith("JES_")
        ),
    )

    def record_jet_systematic(*args):
        calls.append(args)
        return varied_jets

    monkeypatch.setattr(
        analysis_processor, "ApplyJetSystematics", record_jet_systematic
    )

    for systematic in representative_systematics:
        selected_jets = analysis_processor.apply_maintained_jet_systematic(
            year, cleaned_jets, systematic, jet_systematics
        )
        assert selected_jets is varied_jets

    assert calls == [
        (year, cleaned_jets, systematic)
        for systematic in representative_systematics
    ]


def test_nominal_jet_behavior_still_uses_the_jet_helper(monkeypatch):
    calls = []
    cleaned_jets = object()

    def record_jet_systematic(*args):
        calls.append(args)
        return cleaned_jets

    monkeypatch.setattr(
        analysis_processor, "ApplyJetSystematics", record_jet_systematic
    )

    selected_jets = analysis_processor.apply_maintained_jet_systematic(
        "2018", cleaned_jets, "nominal", []
    )

    assert selected_jets is cleaned_jets
    assert calls == [("2018", cleaned_jets, "nominal")]


def test_jet_helper_remains_fail_closed_for_unknown_variations():
    with pytest.raises(Exception, match='Unknown variation "not_a_jet_systematic"'):
        ApplyJetSystematics("2018", object(), "not_a_jet_systematic")
