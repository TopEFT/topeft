import numpy as np
import pytest

from topeft.modules import object_selection as te_os


_PTS = np.array([39.0, 40.0, 41.0, 49.0, 50.0, 51.0])


def _forward_mask(abs_eta, *, apply_eta_band_pt, quality_mask=True):
    eta = np.full_like(_PTS, abs_eta, dtype=float)
    if isinstance(quality_mask, bool):
        quality_mask = np.full_like(_PTS, quality_mask, dtype=bool)
    return te_os.is_forward_jet_eta_banded(
        _PTS,
        eta,
        eta_cut=2.4,
        baseline_pt_cut=40.0,
        apply_eta_band_pt=apply_eta_band_pt,
        eta_band_min=2.5,
        eta_band_max=3.0,
        eta_band_pt_cut=50.0,
        quality_mask=quality_mask,
    ).tolist()


@pytest.mark.parametrize(
    "abs_eta, expected",
    [
        (2.4, [False, False, False, False, False, False]),
        (2.5, [False, False, True, True, True, True]),
        (2.6, [False, False, False, False, False, True]),
        (2.9, [False, False, False, False, False, True]),
        (3.0, [False, False, True, True, True, True]),
        (3.1, [False, False, True, True, True, True]),
        (4.0, [False, False, True, True, True, True]),
    ],
)
def test_forward_eta_banded_policy_enabled_boundary_behavior(abs_eta, expected):
    assert _forward_mask(abs_eta, apply_eta_band_pt=True) == expected


@pytest.mark.parametrize("abs_eta", [2.5, 2.6, 2.9, 3.0, 3.1, 4.0])
def test_forward_eta_banded_policy_disabled_uses_baseline_pt(abs_eta):
    assert _forward_mask(abs_eta, apply_eta_band_pt=False) == [
        False,
        False,
        True,
        True,
        True,
        True,
    ]


def test_forward_eta_banded_policy_disabled_keeps_eta_cut():
    assert _forward_mask(2.4, apply_eta_band_pt=False) == [
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_forward_eta_banded_policy_quality_mask_rejects_passing_jets():
    assert _forward_mask(3.1, apply_eta_band_pt=True, quality_mask=False) == [
        False,
        False,
        False,
        False,
        False,
        False,
    ]


@pytest.mark.parametrize(
    "is_run3, policy, expected",
    [
        (True, "auto", True),
        (False, "auto", False),
        (True, "on", True),
        (False, "on", True),
        (True, "off", False),
        (False, "off", False),
    ],
)
def test_resolve_fwd_eta_band_pt_apply(is_run3, policy, expected):
    assert te_os.resolve_fwd_eta_band_pt_apply(is_run3, policy) is expected


def test_resolve_fwd_eta_band_pt_apply_rejects_invalid_policy():
    with pytest.raises(ValueError, match="Unsupported forward eta-band pT policy"):
        te_os.resolve_fwd_eta_band_pt_apply(True, "sometimes")


def test_resolve_fwd_eta_band_pt_apply_does_not_infer_from_year_strings():
    import inspect

    source = inspect.getsource(te_os.resolve_fwd_eta_band_pt_apply)

    assert 'startswith("201")' not in source
    assert "startswith('201')" not in source
