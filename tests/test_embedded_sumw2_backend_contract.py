"""Backend operations that the embedded-sumw2 migration relies on.

These pin topcoffea's MultiCell HistEFT (``store_sumw2=True``) and Weight-storage
SparseHist behaviour: for a factor c, sumw scales by c and sumw2 by c**2;
independent contributions add their variances; and copy, serialization,
category operations and dense rebinning keep the moments consistent.
"""

import copy
import gzip
import pickle

import cloudpickle
import hist
import numpy as np
import pytest

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist

pytestmark = pytest.mark.skipif(
    not hasattr(hist.storage, "MultiCell"),
    reason="Embedded sumw2 requires boost-histogram MultiCell storage (>=1.7).",
)

WEIGHTS = np.array([2.0, -3.0, 0.5])
POSITIONS = np.array([5.0, 5.0, 25.0])
COEFFS = np.array([[2.0, 1.0, 3.0], [2.0, 7.0, 8.0], [1.0, 0.0, 0.0]])
# Per-bin SM yields and SM-point sumw2 of one fill with the arrays above.
YIELD = np.array([2 * 2 - 3 * 2, 0.0, 0.5, 0.0])
SUMW2 = np.array([(2 * 2) ** 2 + (3 * 2) ** 2, 0.0, 0.25, 0.0])


def _cats():
    return [
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
    ]


def _eft(store=True):
    return HistEFT(
        *_cats(), hist.axis.Variable([0, 10, 20, 30, 40], name="ptz"),
        wc_names=["ctW"], use_multicell=True, store_sumw2=store,
    )


def _filled(process="a"):
    h = _eft()
    h.fill(ptz=POSITIONS, weight=WEIGHTS, eft_coeff=COEFFS, process=process, systematic="nominal")
    return h


def _weight_2d():
    h = SparseHist(
        *_cats(), hist.axis.Regular(2, 0, 2, name="x"), hist.axis.Regular(2, 0, 2, name="y"),
        storage="Weight",
    )
    h.fill(x=[0.5, 0.5, 1.5], y=[0.5, 0.5, 1.5], weight=[2.0, -3.0, 1.0],
           process="a", systematic="nominal")
    return h


def _coefficients(h, flow=False):
    """Per sparse key, the EFT coefficients as (bins, quadratic terms).

    MultiCell views are (cells, bins); the trailing cell is the embedded sumw2.
    """
    n_coeff_cells = lambda v: v.shape[0] - int(h.store_sumw2)
    return {
        key: np.moveaxis(np.asarray(v)[: n_coeff_cells(v)], 0, -1)
        for key, v in h.view(flow=flow).items()
    }


def _moments(h):
    """(sum over sparse bins of SM yield, sum of sumw2) per dense bin."""
    if type(h) is HistEFT:
        yields = sum(v[..., 0] for v in _coefficients(h).values())
        sumw2 = sum(h.nominal_sumw2().values())
    else:
        views = list(h.view(as_dict=True).values())
        yields = sum(np.asarray(v.value) for v in views)
        sumw2 = sum(np.asarray(v.variance) for v in views)
    return np.asarray(yields), np.asarray(sumw2)


def test_fill_records_sm_point_second_moment():
    yields, sumw2 = _moments(_filled())
    np.testing.assert_allclose(yields, YIELD)
    np.testing.assert_allclose(sumw2, SUMW2)


def test_fill_sumw2_false_leaves_slot_unfilled():
    h = _eft()
    h.fill(ptz=POSITIONS, weight=WEIGHTS, eft_coeff=COEFFS, process="a",
           systematic="JESUp", fill_sumw2=False)
    yields, sumw2 = _moments(h)
    np.testing.assert_allclose(yields, YIELD)
    assert not sumw2.any()


@pytest.mark.parametrize(
    "factor, apply",
    [
        (-1.0, lambda h, c: h.scale(c)),
        (-2.0, lambda h, c: h * c),
        (-2.0, lambda h, c: c * h),
        (np.float32(-2.0), lambda h, c: h * c),
        (0.5, lambda h, c: h / (1 / c)),
    ],
)
@pytest.mark.parametrize("make", [_filled, _weight_2d])
def test_scaling_is_linear_in_sumw_and_quadratic_in_sumw2(make, factor, apply):
    base_yield, base_sumw2 = _moments(make())
    yields, sumw2 = _moments(apply(make(), factor))
    np.testing.assert_allclose(yields, factor * base_yield)
    np.testing.assert_allclose(sumw2, factor ** 2 * base_sumw2)
    assert np.all(sumw2 >= 0)


def test_bin_dependent_factors_are_refused():
    with pytest.raises(TypeError, match="real numeric scalar"):
        _filled() * np.array([1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize("make", [_filled, _weight_2d])
def test_direct_subtraction_is_refused(make):
    with pytest.raises((NotImplementedError, AttributeError, TypeError)):
        make() - make()


@pytest.mark.parametrize("make", [_filled, _weight_2d])
def test_prompt_subtraction_pattern_adds_variances(make):
    """Nonprompt pattern: hPromptSub.scale(-1); hFakes += hPromptSub."""
    fakes, prompt = make(), make().scale(0.5)
    fakes_yield, fakes_sumw2 = _moments(fakes)
    prompt_yield, prompt_sumw2 = _moments(prompt)
    fakes += prompt.scale(-1)
    yields, sumw2 = _moments(fakes)
    np.testing.assert_allclose(yields, fakes_yield - prompt_yield)
    np.testing.assert_allclose(sumw2, fakes_sumw2 + prompt_sumw2)


def test_addition_merges_sparse_bins_and_adds_moments():
    merged = _filled("a") + _filled("b")
    merged += _filled("a")
    per_process = {
        tuple(k)[0]: v for k, v in merged.nominal_sumw2().items()
    }
    np.testing.assert_allclose(per_process["a"], 2 * SUMW2)
    np.testing.assert_allclose(per_process["b"], SUMW2)


def test_layout_mismatch_is_refused():
    with pytest.raises(ValueError, match="matching EFT storage"):
        _filled() + _eft(store=False)
    # Weight + Double SparseHist mixing is not guarded by topcoffea; a payload
    # has one storage mode, enforced by nominal_schema.validate_embedded_nominal_mapping.


@pytest.mark.parametrize(
    "roundtrip",
    [
        copy.deepcopy,
        lambda h: h.copy(),
        lambda h: pickle.loads(pickle.dumps(h)),
        lambda h: pickle.loads(gzip.decompress(gzip.compress(cloudpickle.dumps(h)))),
    ],
)
@pytest.mark.parametrize("make", [_filled, _weight_2d])
def test_copy_and_serialization_preserve_moments(make, roundtrip):
    original = make()
    restored = roundtrip(original)
    for left, right in zip(_moments(original), _moments(restored)):
        np.testing.assert_allclose(left, right)
    if type(original) is HistEFT:
        assert restored.store_sumw2


def test_shallow_copy_is_an_empty_histogram():
    """SparseHist.__copy__ returns an empty clone; never use it to copy content."""
    shallow = copy.copy(_filled())
    assert shallow.empty() and shallow.store_sumw2


@pytest.mark.parametrize(
    "operation",
    [
        lambda h: h.integrate("process"),
        lambda h: h.integrate("systematic", "nominal"),
        lambda h: h.group("process", {"ab": ["a", "b"]}),
        lambda h: h.remove("process", ["b"]) + h.remove("process", ["a"]),
        lambda h: h[{"process": ["a", "b"]}],
    ],
)
def test_category_operations_preserve_moment_totals(operation):
    h = _filled("a") + _filled("b")
    expected = [m.sum() for m in _moments(h)]
    result = [m.sum() for m in _moments(operation(h))]
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("make", [_filled, _weight_2d])
def test_hist_rebin_adds_moments(make):
    h = make()
    axis = "ptz" if type(h) is HistEFT else "x"
    rebinned = h[{axis: hist.rebin(2)}]
    for before, after in zip(_moments(h), _moments(rebinned)):
        np.testing.assert_allclose(after.sum(), before.sum())


def test_dense_slice_keeps_selected_bins():
    sliced = _filled()[{"ptz": slice(2, 4)}]
    np.testing.assert_allclose(_moments(sliced)[1], SUMW2[2:4])


def test_sm_evaluation_ignores_the_sumw2_slot():
    evaluated = _filled().eval({})
    (values,) = evaluated.values()
    np.testing.assert_allclose(values[1:-1], YIELD)


@pytest.mark.parametrize("store", [True, False])
def test_topeft_semantic_rebin_supports_multicell(store):
    """axis_binning.rebin_histogram aggregates the bin axis of (cells, bins) views."""
    from topeft.modules.axis_binning import rebin_histogram

    h = _eft(store=store)
    h.fill(ptz=np.array([5.0, 15.0, 25.0, 35.0, 99.0]), weight=np.array([2.0, -3.0, 0.5, 1.0, 4.0]),
           eft_coeff=np.array([[2.0, 1.0, 3.0], [2.0, 7.0, 8.0], [1.0, 0.0, 0.0],
                               [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
           process="a", systematic="nominal")
    rebinned = rebin_histogram(h, [0, 20, 40])
    assert rebinned.store_sumw2 is store
    ((key, source),) = _coefficients(h, flow=True).items()
    ((_, target),) = _coefficients(rebinned, flow=True).items()
    # flow bins: [under, 0-10, 10-20, 20-30, 30-40, over] -> [under, 0-20, 20-40, over]
    np.testing.assert_allclose(target[0], source[0])
    np.testing.assert_allclose(target[1], source[1] + source[2])
    np.testing.assert_allclose(target[2], source[3] + source[4])
    np.testing.assert_allclose(target[3], source[5])
    if store:
        (source_w2,) = h.nominal_sumw2(flow=True).values()
        (target_w2,) = rebinned.nominal_sumw2(flow=True).values()
        np.testing.assert_allclose(
            target_w2, [source_w2[0], source_w2[1] + source_w2[2],
                        source_w2[3] + source_w2[4], source_w2[5]],
        )
        np.testing.assert_allclose(target_w2.sum(), source_w2.sum())


def test_topeft_semantic_rebin_supports_weight_storage():
    from topeft.modules.axis_binning import rebin_histogram

    h = SparseHist(*_cats(), hist.axis.Variable([0, 10, 20, 30, 40], name="ptz"), storage="Weight")
    h.fill(ptz=np.array([5.0, 15.0, 25.0]), weight=np.array([2.0, -3.0, 0.5]),
           process="a", systematic="nominal")
    rebinned = rebin_histogram(h, [0, 20, 40])
    (view,) = rebinned.view(flow=True, as_dict=True).values()
    np.testing.assert_allclose(np.asarray(view.value), [0.0, -1.0, 0.5, 0.0])
    np.testing.assert_allclose(np.asarray(view.variance), [0.0, 13.0, 0.25, 0.0])
