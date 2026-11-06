"""Schema version 3 contract: embedded nominal sumw2 (EMBEDDED / NONE only)."""

import copy
import pickle

import hist
import numpy as np
import pytest

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist

from topeft.modules import nominal_schema as ns

pytestmark = pytest.mark.skipif(
    not hasattr(hist.storage, "MultiCell"),
    reason="Embedded sumw2 requires boost-histogram MultiCell storage (>=1.7).",
)

WCS = ["ctW", "ctG"]
FAMILY_1D = "ptz"
FAMILY_2D = "lepton_pt_vs_eta"
FAMILIES = (FAMILY_1D, FAMILY_2D)


def _cats():
    return [
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
    ]


def _dense_1d():
    return hist.axis.Variable([0, 50, 100, 200], name=FAMILY_1D, label="ptz")


def _dense_2d():
    return [
        hist.axis.Regular(2, 0, 100, name="lepton_pt"),
        hist.axis.Regular(2, -2.5, 2.5, name="lepton_eta"),
    ]


def _scalar(store=True):
    return HistEFT(*_cats(), _dense_1d(), wc_names=[], use_multicell=True, store_sumw2=store)


def _eft(store=True, wcs=WCS):
    return HistEFT(*_cats(), _dense_1d(), wc_names=wcs, use_multicell=True, store_sumw2=store)


def _two_d(storage="Weight"):
    return SparseHist(*_cats(), *_dense_2d(), storage=storage)


def _payload(store=True):
    return {
        ns.scalar_nominal_key(FAMILY_1D): _scalar(store),
        ns.eft_nominal_key(FAMILY_1D): _eft(store),
        FAMILY_2D: _two_d("Weight" if store else "Double"),
    }


def _fill_1d(histogram, *, process, systematic, weights, coeffs=None, fill_sumw2=None):
    if fill_sumw2 is None:
        fill_sumw2 = systematic == ns.SUMW2_RECORDED_SYSTEMATIC
    histogram.fill(
        ptz=np.full(len(weights), 25.0),
        weight=np.asarray(weights, dtype=float),
        eft_coeff=coeffs,
        process=process,
        channel="3l",
        systematic=systematic,
        appl="isSR",
        fill_sumw2=fill_sumw2,
    )


# --- storage-mode parsing -------------------------------------------------


@pytest.mark.parametrize("value", ["embedded", "EMBEDDED", " Embedded ", ns.Sumw2Storage.EMBEDDED])
def test_parse_embedded(value):
    assert ns.parse_sumw2_storage(value) is ns.Sumw2Storage.EMBEDDED


def test_parse_none_and_fill_flag():
    assert ns.parse_sumw2_storage("none") is ns.Sumw2Storage.NONE
    assert ns.sumw2_storage_from_fill_flag(True) is ns.Sumw2Storage.EMBEDDED
    assert ns.sumw2_storage_from_fill_flag(False) is ns.Sumw2Storage.NONE
    with pytest.raises(TypeError):
        ns.sumw2_storage_from_fill_flag("yes")


def test_companion_storage_is_rejected_by_name():
    with pytest.raises(ValueError, match="Companion"):
        ns.parse_sumw2_storage("companion")
    with pytest.raises(ValueError, match="Unknown sumw2 storage"):
        ns.parse_sumw2_storage("production")
    with pytest.raises(ValueError, match="Unknown sumw2 storage"):
        ns.parse_sumw2_storage(None)


# --- payload validation ----------------------------------------------------


@pytest.mark.parametrize("store, expected", [(True, "embedded"), (False, "none")])
def test_valid_payloads_report_their_storage(store, expected):
    payload = _payload(store)
    storage = ns.validate_embedded_nominal_mapping(payload, runtime_families=FAMILIES)
    assert storage is ns.parse_sumw2_storage(expected)
    assert ns.validate_embedded_nominal_mapping(
        payload, runtime_families=FAMILIES, sumw2_storage=expected
    ) is storage


def test_companion_keys_are_rejected():
    payload = _payload()
    payload[ns.sumw2_key(FAMILY_1D)] = SparseHist(*_cats(), _dense_1d(), storage="Double")
    with pytest.raises(ValueError, match="Companion sumw2 histograms are not supported"):
        ns.validate_embedded_nominal_mapping(payload, runtime_families=FAMILIES)
    with pytest.raises(ValueError, match="not supported by schema version 3"):
        ns.validate_embedded_nominal_family(payload, FAMILY_1D)


def test_mixed_storage_across_families_is_rejected():
    payload = _payload(True)
    payload[FAMILY_2D] = _two_d("Double")
    with pytest.raises(ValueError, match="mixes sumw2 storage modes"):
        ns.validate_embedded_nominal_mapping(payload, runtime_families=FAMILIES)


def test_mixed_storage_within_family_is_rejected():
    payload = _payload(True)
    payload[ns.eft_nominal_key(FAMILY_1D)] = _eft(store=False)
    with pytest.raises(ValueError, match="mixes embedded and none"):
        ns.validate_embedded_nominal_mapping(payload, runtime_families=FAMILIES)


def test_declared_storage_must_match_content():
    with pytest.raises(ValueError, match="disagrees with the histogram content"):
        ns.validate_embedded_nominal_mapping(
            _payload(True), runtime_families=FAMILIES, sumw2_storage="none"
        )


def test_empty_payload_requires_declaration():
    with pytest.raises(ValueError, match="Cannot infer"):
        ns.validate_embedded_nominal_mapping({}, runtime_families=FAMILIES)
    assert ns.validate_embedded_nominal_mapping(
        {}, runtime_families=FAMILIES, sumw2_storage="none"
    ) is ns.Sumw2Storage.NONE


@pytest.mark.parametrize(
    "key, replacement, message",
    [
        (ns.scalar_nominal_key(FAMILY_1D),
         lambda: SparseHist(*_cats(), _dense_1d(), storage="Double"),
         "must be an exact MultiCell HistEFT"),
        (ns.eft_nominal_key(FAMILY_1D),
         lambda: HistEFT(*_cats(), _dense_1d(), wc_names=WCS, use_multicell=False),
         "legacy quadratic_term layout"),
        (ns.scalar_nominal_key(FAMILY_1D), lambda: _eft(), "must have no Wilson coefficients"),
        (ns.eft_nominal_key(FAMILY_1D), lambda: _scalar(), "must declare its Wilson coefficients"),
        (FAMILY_2D, lambda: SparseHist(*_cats(), *_dense_2d(), storage="Int64"),
         "storage='Weight' or storage='Double'"),
    ],
)
def test_noncompliant_components_are_rejected(key, replacement, message):
    payload = _payload(True)
    payload[key] = replacement()
    with pytest.raises((TypeError, ValueError), match=message):
        ns.validate_embedded_nominal_mapping(payload, runtime_families=FAMILIES)


def test_legacy_and_orphan_keys_are_rejected():
    payload = _payload(True)
    payload[FAMILY_1D] = _scalar()
    with pytest.raises(ValueError, match="duplicate authoritative"):
        ns.validate_embedded_nominal_mapping(payload, runtime_families=FAMILIES)
    payload = _payload(True)
    payload[ns.scalar_nominal_key("njets")] = _scalar()
    with pytest.raises(ValueError, match="orphan or unresolved"):
        ns.validate_embedded_nominal_mapping(payload, runtime_families=FAMILIES)


def test_sibling_axes_must_agree():
    payload = _payload(True)
    payload[ns.eft_nominal_key(FAMILY_1D)] = HistEFT(
        *_cats(), hist.axis.Variable([0, 100, 200], name=FAMILY_1D),
        wc_names=WCS, use_multicell=True, store_sumw2=True,
    )
    with pytest.raises(ValueError, match="different dense axes"):
        ns.validate_embedded_nominal_mapping(payload, runtime_families=FAMILIES)


def test_physical_storage_must_match_declared_storage():
    histogram = _two_d("Weight")
    histogram.fill(lepton_pt=[10.0], lepton_eta=[0.0], process="p", channel="c",
                   systematic="nominal", appl="isSR")
    (index,) = histogram._dense_hists
    histogram._dense_hists[index] = hist.Hist(*_dense_2d(), storage="Double")
    with pytest.raises(TypeError, match="declares 'weight' storage"):
        ns.sumw2_storage_of_histogram(histogram, key=FAMILY_2D)


def test_serialized_payload_still_validates():
    payload = _payload(True)
    _fill_1d(payload[ns.scalar_nominal_key(FAMILY_1D)], process="ttZ",
             systematic="nominal", weights=[1.5])
    restored = pickle.loads(pickle.dumps(payload))
    assert ns.validate_embedded_nominal_mapping(
        restored, runtime_families=FAMILIES
    ) is ns.Sumw2Storage.EMBEDDED


def test_applicability_structure_uses_split_layout():
    payload = _payload(True)
    _fill_1d(payload[ns.scalar_nominal_key(FAMILY_1D)], process="ttZ",
             systematic="nominal", weights=[1.0])
    applicability = {"families": {FAMILY_1D: {"channels": {"2lss": "not_applicable", "3l": "applicable"}}}}
    ns.validate_embedded_nominal_mapping(
        {k: v for k, v in payload.items() if k != FAMILY_2D},
        runtime_families=(FAMILY_1D,),
        histogram_applicability=applicability,
    )
    applicability["families"][FAMILY_1D]["channels"]["4l"] = "applicable"
    with pytest.raises(ValueError, match="structurally absent"):
        ns.validate_embedded_nominal_mapping(
            {k: v for k, v in payload.items() if k != FAMILY_2D},
            runtime_families=(FAMILY_1D,),
            histogram_applicability=applicability,
        )


# --- recorded nominal sumw2 ------------------------------------------------


def _values_by_process(moments, histogram):
    names = list(histogram.categorical_axes.name)
    return {
        tuple(key)[names.index("process")]: np.asarray(values)
        for key, values in moments.items()
    }


def test_scalar_mc_data_and_eft_sm_point_moments():
    scalar = _scalar()
    _fill_1d(scalar, process="ttZ", systematic="nominal", weights=[2.0, -3.0])
    _fill_1d(scalar, process="data", systematic="nominal", weights=[1.0, 1.0, 1.0])
    eft = _eft()
    coeffs = np.array([[2.0, 1.0, 3.0, 4.0, 5.0, 6.0], [2.0, 7.0, 8.0, 9.0, 10.0, 11.0]])
    _fill_1d(eft, process="ttH", systematic="nominal", weights=[2.0, -3.0], coeffs=coeffs)

    scalar_moments = _values_by_process(ns.get_nominal_sumw2(scalar), scalar)
    assert scalar_moments["ttZ"].sum() == pytest.approx(4.0 + 9.0)
    assert scalar_moments["data"].sum() == pytest.approx(3.0)  # unit weights -> count
    eft_moments = _values_by_process(ns.get_nominal_sumw2(eft), eft)
    # SM point only: (2*2)**2 + (-3*2)**2; WC-dependent terms never enter.
    assert eft_moments["ttH"].sum() == pytest.approx(16.0 + 36.0)


def test_non_nominal_bins_are_not_recorded_rather_than_zero():
    scalar = _scalar()
    _fill_1d(scalar, process="ttZ", systematic="nominal", weights=[2.0])
    _fill_1d(scalar, process="ttZ", systematic="JESUp", weights=[3.0])
    two_d = _two_d("Weight")
    for systematic, weight in (("nominal", 2.0), ("JESUp", 3.0)):
        two_d.fill(lepton_pt=[10.0], lepton_eta=[0.0], weight=[weight], process="ttZ",
                   channel="3l", systematic=systematic, appl="isSR")

    for histogram in (scalar, two_d):
        moments = ns.get_nominal_sumw2(histogram)
        systematics = {tuple(k)[2] for k in moments}
        assert systematics == {"nominal"}
        assert sum(np.sum(v) for v in moments.values()) == pytest.approx(4.0)
        for key in histogram.view(as_dict=True):
            assert ns.is_nominal_sumw2_recorded(histogram, key) is (tuple(key)[2] == "nominal")
    # Weight storage accumulates w**2 in the JESUp bin; it is still unrecorded.
    jes_variance = [
        np.asarray(v.variance).sum()
        for k, v in two_d.view(as_dict=True).items() if tuple(k)[2] == "JESUp"
    ]
    assert jes_variance == [pytest.approx(9.0)]


def test_histogram_without_systematic_axis_is_nominal_content():
    scalar = _scalar()
    _fill_1d(scalar, process="ttZ", systematic="nominal", weights=[2.0])
    nominal_slice = scalar[{"systematic": "nominal"}]
    assert "systematic" not in nominal_slice.categorical_axes.name
    moments = ns.get_nominal_sumw2(nominal_slice)
    assert sum(np.sum(v) for v in moments.values()) == pytest.approx(4.0)


def test_none_storage_has_no_sumw2():
    payload = _payload(False)
    for histogram in payload.values():
        with pytest.raises(ns.Sumw2NotStoredError):
            ns.get_nominal_sumw2(histogram)
    with pytest.raises(ns.Sumw2NotStoredError):
        ns.require_embedded_sumw2(payload, runtime_families=FAMILIES)
    ns.require_embedded_sumw2(_payload(True), runtime_families=FAMILIES)


def test_returned_moments_are_copies():
    scalar = _scalar()
    _fill_1d(scalar, process="ttZ", systematic="nominal", weights=[2.0])
    moments = ns.get_nominal_sumw2(scalar)
    for values in moments.values():
        values[...] = -1.0
    assert all(np.all(v >= 0) for v in ns.get_nominal_sumw2(scalar).values())
    assert ns.get_nominal_sumw2(copy.deepcopy(scalar)).keys() == moments.keys()
