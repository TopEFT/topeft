from __future__ import annotations

import pickle

import pytest

hist = pytest.importorskip("hist")
np = pytest.importorskip("numpy")

from topcoffea.modules.histEFT import HistEFT


def _make_histeft(dense_name: str = "x", wc_names: list[str] | None = None) -> HistEFT:
    return HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.Regular(2, 0.0, 2.0, name=dense_name),
        wc_names=wc_names or ["ctG", "cpt"],
        label="Events",
    )


def _only_eval_array(histo: HistEFT, wc_values: dict[str, float] | None = None) -> np.ndarray:
    evaluated = histo.eval({} if wc_values is None else wc_values)
    assert len(evaluated) == 1
    return np.asarray(next(iter(evaluated.values())), dtype=float)


def test_quadratic_term_order_matches_current_lower_triangle_contract():
    histo = _make_histeft()

    assert histo.wc_names == ["ctG", "cpt"]
    assert histo.quadratic_term_index("sm", "sm") == 0
    assert histo.quadratic_term_index("ctG", "sm") == 1
    assert histo.quadratic_term_index("ctG", "ctG") == 2
    assert histo.quadratic_term_index("cpt", "sm") == 3
    assert histo.quadratic_term_index("cpt", "ctG") == 4
    assert histo.quadratic_term_index("cpt", "cpt") == 5


def test_fill_weights_and_eval_match_two_wc_polynomial_contract():
    histo = _make_histeft()
    coeffs = np.asarray(
        [
            [1.0, 2.0, 3.0, 5.0, 7.0, 11.0],
            [4.0, -1.0, 0.5, 3.0, -2.0, 0.25],
        ]
    )

    histo.fill(
        process="ttH",
        channel="2lss",
        systematic="nominal",
        appl="isSR",
        x=np.asarray([0.25, 1.25]),
        weight=np.asarray([2.0, -1.0]),
        eft_coeff=coeffs,
    )

    np.testing.assert_allclose(_only_eval_array(histo), [0.0, 2.0, -4.0, 0.0])
    np.testing.assert_allclose(
        _only_eval_array(histo, {"ctG": 1.0, "cpt": 2.0}),
        [0.0, 148.0, -6.5, 0.0],
    )


def test_missing_eft_coeff_fills_sm_only_and_unknown_wc_raises():
    histo = _make_histeft()
    histo.fill(
        process="background",
        channel="3l",
        systematic="nominal",
        appl="isCR",
        x=np.asarray([0.25]),
        weight=np.asarray([3.0]),
    )

    np.testing.assert_allclose(_only_eval_array(histo), [0.0, 3.0, 0.0, 0.0])
    np.testing.assert_allclose(
        _only_eval_array(histo, {"ctG": 2.0, "cpt": -4.0}),
        [0.0, 3.0, 0.0, 0.0],
    )
    with pytest.raises(LookupError, match="does not know about"):
        histo.eval({"unknown_wc": 1.0})


def test_integrate_group_copy_add_and_as_hist_cover_consumer_operations():
    histo = _make_histeft(wc_names=[])
    for process, x_value, weight in (
        ("a", 0.25, 1.0),
        ("b", 1.25, 2.0),
        ("c", 1.25, 4.0),
    ):
        histo.fill(
            process=process,
            channel="2lss",
            systematic="nominal",
            appl="isSR",
            x=np.asarray([x_value]),
            weight=np.asarray([weight]),
        )

    nominal = histo.integrate("systematic", "nominal")
    grouped = nominal.group("process", {"combo": ["a", "b"]})
    grouped_dense = grouped.as_hist({})

    np.testing.assert_allclose(
        grouped_dense[
            {"process": "combo", "channel": "2lss", "appl": "isSR"}
        ].values(flow=True),
        [0.0, 1.0, 2.0, 0.0],
    )

    copied = grouped.copy()
    copied += grouped
    np.testing.assert_allclose(
        copied.as_hist({})[
            {"process": "combo", "channel": "2lss", "appl": "isSR"}
        ].values(flow=True),
        [0.0, 2.0, 4.0, 0.0],
    )


def test_pickle_round_trip_preserves_wc_metadata_and_sumw2_companion_shape():
    histo = _make_histeft()
    sumw2 = _make_histeft(dense_name="x_sumw2")
    coeffs = np.asarray([[1.0, 0.5, 0.25, -1.0, 0.0, 2.0]])

    fill_payload = {
        "process": "ttH",
        "channel": "2lss",
        "systematic": "nominal",
        "appl": "isSR",
        "x": np.asarray([0.25]),
        "weight": np.asarray([2.0]),
        "eft_coeff": coeffs,
    }
    histo.fill(**fill_payload)

    sumw2_payload = dict(fill_payload)
    sumw2_payload.pop("x")
    sumw2_payload["x_sumw2"] = np.asarray([0.25])
    sumw2_payload["weight"] = np.square(fill_payload["weight"])
    sumw2.fill(**sumw2_payload)

    restored = pickle.loads(pickle.dumps({"x": histo, "x_sumw2": sumw2}))

    assert set(restored) == {"x", "x_sumw2"}
    assert restored["x"].wc_names == ["ctG", "cpt"]
    assert restored["x_sumw2"].wc_names == ["ctG", "cpt"]
    np.testing.assert_allclose(_only_eval_array(restored["x"]), [0.0, 2.0, 0.0, 0.0])
    np.testing.assert_allclose(
        _only_eval_array(restored["x_sumw2"]),
        [0.0, 4.0, 0.0, 0.0],
    )
