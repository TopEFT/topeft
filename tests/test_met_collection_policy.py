from types import SimpleNamespace

import pytest

from topeft.modules.corrections import (
    ApplyMETSystematics,
    get_selected_met,
    get_supported_met_systematics,
    is_met_unclustered_systematic,
)


def test_run2_selected_met_uses_legacy_met():
    events = SimpleNamespace(MET=object(), PuppiMET=object())

    assert get_selected_met(events, "2018") is events.MET


def test_run3_selected_met_uses_puppimet():
    events = SimpleNamespace(MET=object(), PuppiMET=object())

    assert get_selected_met(events, "2022") is events.PuppiMET


def test_run3_selected_met_missing_puppimet_fails_clearly():
    events = SimpleNamespace(MET=object())

    with pytest.raises(RuntimeError, match="requires events.PuppiMET"):
        get_selected_met(events, "2022")


def test_met_unclustered_systematics_are_public_generic_labels():
    assert get_supported_met_systematics("2022", isData=False) == [
        "MET_UnclusteredEnergyUp",
        "MET_UnclusteredEnergyDown",
    ]
    assert get_supported_met_systematics("2022", isData=True) == []
    assert is_met_unclustered_systematic("MET_UnclusteredEnergyUp")
    assert is_met_unclustered_systematic("MET_UnclusteredEnergyDown")
    assert not is_met_unclustered_systematic("JER_2022Up")


def test_apply_met_systematics_selects_unclustered_shift_only():
    nominal = object()
    up = object()
    down = object()
    met = SimpleNamespace(
        MET_UnclusteredEnergy=SimpleNamespace(up=up, down=down)
    )

    assert ApplyMETSystematics(met, "nominal") is met
    assert ApplyMETSystematics(met, "MET_UnclusteredEnergyUp") is up
    assert ApplyMETSystematics(met, "MET_UnclusteredEnergyDown") is down
    assert ApplyMETSystematics(nominal, "JER_2022Up") is nominal
