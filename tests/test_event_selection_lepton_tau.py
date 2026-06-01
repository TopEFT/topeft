import awkward as ak
import pytest
from coffea.nanoevents.methods import candidate

from topeft.modules import event_selection as es


ak.behavior.update(candidate.behavior)


def _objects(items):
    return ak.with_name(ak.Array([items]), "PtEtaPhiMCandidate")


def _event(leptons, tau=None):
    taus = _objects([tau or _tau()])
    padded_leptons = ak.pad_none(_objects(leptons), 3)
    return padded_leptons[:, 0], padded_leptons[:, 1], padded_leptons[:, 2], taus[:, 0]


def _lep(pt=10.0, pdg_id=-11):
    return {"pt": pt, "eta": 0.0, "phi": 0.0, "mass": 0.0, "pdgId": pdg_id, "charge": -1 if pdg_id > 0 else 1}


def _tau(charge=-1):
    return {"pt": 80.0, "eta": 0.0, "phi": 3.14159265, "mass": 0.0, "charge": charge}


def _pt(value):
    return value.tolist()[0]


def _is_none(value):
    return value.tolist()[0] is None


def _is_not_none(value):
    return value.tolist()[0] is not None


def test_get_Zlt_pt_returns_l0_tau_pt_when_l0_passes_and_l1_fails():
    l0, l1, _, tau0 = _event([_lep(pdg_id=-11), _lep(pdg_id=11)])

    result = es.get_Zlt_pt(l0, l1, tau0)

    assert _pt(result) == pytest.approx(_pt((l0 + tau0).pt))


def test_get_Zlt_pt_returns_l1_tau_pt_when_l1_passes_and_l0_fails():
    l0, l1, _, tau0 = _event([_lep(pdg_id=11), _lep(pdg_id=-11)])

    result = es.get_Zlt_pt(l0, l1, tau0)

    assert _pt(result) == pytest.approx(_pt((l1 + tau0).pt))


def test_get_Zlt_pt_preserves_l0_priority_when_both_candidates_pass():
    l0, l1, _, tau0 = _event([_lep(pt=10.0, pdg_id=-11), _lep(pt=20.0, pdg_id=-11)])

    result = es.get_Zlt_pt(l0, l1, tau0)

    assert _pt(result) == pytest.approx(_pt((l0 + tau0).pt))


def test_get_Zlt_pt_handles_missing_l1_when_l0_tau_passes():
    l0, l1, _, tau0 = _event([_lep(pdg_id=-11)])

    result = es.get_Zlt_pt(l0, l1, tau0)

    assert _pt(result) == pytest.approx(_pt((l0 + tau0).pt))


def test_get_Zlt_pt_returns_none_when_l0_fails_and_l1_is_missing():
    l0, l1, _, tau0 = _event([_lep(pdg_id=11)])

    result = es.get_Zlt_pt(l0, l1, tau0)

    assert _is_none(result)


def test_get_Zlt_pt_returns_none_when_neither_l0_nor_l1_passes():
    l0, l1, _, tau0 = _event([_lep(pdg_id=11), _lep(pdg_id=13)])

    result = es.get_Zlt_pt(l0, l1, tau0)

    assert _is_none(result)


def test_get_Zlt_pt_ignores_l2_even_if_l2_tau_would_pass():
    l0, l1, l2, tau0 = _event([_lep(pdg_id=11), _lep(pdg_id=13), _lep(pdg_id=-11)])

    result = es.get_Zlt_pt(l0, l1, tau0)

    assert _is_none(result)
    assert bool(_pt(es.lt_Z_mask(l2, l1, tau0))) is True


@pytest.mark.parametrize(
    ("leptons", "expected"),
    [
        ([_lep(pdg_id=-11), _lep(pdg_id=11)], True),
        ([_lep(pdg_id=11), _lep(pdg_id=-11)], True),
        ([_lep(pt=10.0, pdg_id=-11), _lep(pt=20.0, pdg_id=-11)], True),
        ([_lep(pdg_id=11), _lep(pdg_id=13)], False),
        ([_lep(pdg_id=-11)], True),
        ([_lep(pdg_id=11), _lep(pdg_id=13), _lep(pdg_id=-11)], False),
    ],
)
def test_lt_Z_mask_is_or_of_explicit_l0_l1_tau_candidates(leptons, expected):
    l0, l1, _, tau0 = _event(leptons)

    result = es.lt_Z_mask(l0, l1, tau0)

    assert bool(_pt(result)) is expected


@pytest.mark.parametrize(
    ("lep", "tau", "expected"),
    [
        (_lep(pdg_id=11), _tau(charge=1), True),
        (_lep(pdg_id=-11), _tau(charge=-1), True),
        (_lep(pdg_id=11), _tau(charge=-1), False),
        (_lep(pdg_id=-11), _tau(charge=1), False),
    ],
)
def test_lt_Z_mask_charge_convention_is_opposite_physical_charge(lep, tau, expected):
    l0, l1, _, tau0 = _event([lep], tau=tau)

    result = es.lt_Z_mask(l0, l1, tau0)

    assert bool(_pt(result)) is expected


@pytest.mark.parametrize(
    "leptons",
    [
        [_lep(pdg_id=-11), _lep(pdg_id=11)],
        [_lep(pdg_id=11), _lep(pdg_id=-11)],
        [_lep(pt=10.0, pdg_id=-11), _lep(pt=20.0, pdg_id=-11)],
        [_lep(pdg_id=11), _lep(pdg_id=13)],
        [_lep(pdg_id=-11)],
        [_lep(pdg_id=11)],
        [_lep(pdg_id=11), _lep(pdg_id=13), _lep(pdg_id=-11)],
    ],
)
def test_lt_Z_mask_and_get_Zlt_pt_are_coherent(leptons):
    l0, l1, _, tau0 = _event(leptons)

    tau_on_z_mask = es.lt_Z_mask(l0, l1, tau0)
    ptz_wtau = es.get_Zlt_pt(l0, l1, tau0)

    if bool(_pt(tau_on_z_mask)):
        assert _is_not_none(ptz_wtau)
    else:
        assert _is_none(ptz_wtau)
