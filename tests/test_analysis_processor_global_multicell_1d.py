"""AnalysisProcessor storage contract for embedded nominal sumw2 (schema v3)."""

import pickle

import hist
import numpy as np
import pytest

from analysis.topeft_run2.analysis_processor import AnalysisProcessor
from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules import nominal_schema as ns
from topeft.modules.axis_binning import rebin_histogram

pytestmark = pytest.mark.skipif(
    not hasattr(hist.storage, "MultiCell"),
    reason="Embedded sumw2 requires boost-histogram MultiCell storage (>=1.7).",
)

WCS = ["ctW", "ctG"]
HISTS = ["ptz", "njets", "lepton_pt_vs_eta"]
FAMILIES = ("ptz", "njets", "lepton_pt_vs_eta")


def _processor(fill_sumw2_hist=True, record_raw_count=True):
    return AnalysisProcessor(
        samples={},
        wc_names_lst=WCS,
        hist_lst=HISTS,
        fill_sumw2_hist=fill_sumw2_hist,
        record_raw_count=record_raw_count,
    )


@pytest.mark.parametrize("fill_sumw2_hist", [True, False])
def test_allocation_uses_one_storage_mode_without_companions(fill_sumw2_hist):
    processor = _processor(fill_sumw2_hist)
    accumulator = processor.accumulator
    expected_storage = ns.sumw2_storage_from_fill_flag(fill_sumw2_hist)

    assert processor.sumw2_storage is expected_storage
    assert processor.nominal_container_schema_version == ns.EMBEDDED_SUMW2_SCHEMA_VERSION
    assert processor.nominal_container_layout == ns.EMBEDDED_SUMW2_LAYOUT
    assert not any(key.endswith("_sumw2") for key in accumulator)
    for family in ("ptz", "njets"):
        scalar = accumulator[ns.scalar_nominal_key(family)]
        eft = accumulator[ns.eft_nominal_key(family)]
        for histogram in (scalar, eft):
            assert type(histogram) is HistEFT
            assert histogram._use_multicell
            assert histogram.store_sumw2 is fill_sumw2_hist
        assert scalar.wc_names == [] and eft.wc_names == WCS
    two_d = accumulator["lepton_pt_vs_eta"]
    assert type(two_d) is SparseHist
    assert two_d._init_args["storage"] == ("Weight" if fill_sumw2_hist else "Double")
    assert ns.validate_embedded_nominal_mapping(
        accumulator, runtime_families=FAMILIES, sumw2_storage=expected_storage
    ) is expected_storage


def test_raw_count_tracking_follows_fitting_families_on_multicell():
    accumulator = _processor().accumulator
    assert accumulator[ns.scalar_nominal_key("ptz")].track_raw_counts
    assert accumulator[ns.eft_nominal_key("ptz")].track_raw_counts
    assert not accumulator[ns.scalar_nominal_key("njets")].track_raw_counts
    assert not accumulator["lepton_pt_vs_eta"].track_raw_counts


def test_selective_policy_and_non_boolean_flag_are_rejected():
    with pytest.raises(ValueError, match="sumw2_policy is disabled"):
        AnalysisProcessor(samples={}, hist_lst=["njets"], sumw2_policy=object())
    with pytest.raises(TypeError, match="fill_sumw2 must be a boolean"):
        AnalysisProcessor(samples={}, hist_lst=["njets"], fill_sumw2_hist=1)


def _fill_1d(processor, histogram, *, process, systematic, is_data, coeffs=None):
    info = dict(
        ptz=np.asarray([25.0, 25.0]), weight=np.asarray([2.0, -3.0]),
        process=process, channel="3l", appl="isSR", systematic=systematic,
    )
    if coeffs is not None:
        info["eft_coeff"] = coeffs
    processor._fill_nominal_histogram(histogram, info, is_data=is_data, wgt_fluct=systematic)


def _by_key(moments, names):
    return {tuple(k)[names.index("process")] + "/" + tuple(k)[names.index("systematic")]: float(np.sum(v))
            for k, v in moments.items()}


def test_fill_records_nominal_sumw2_for_data_mc_and_eft_sm_point():
    processor = _processor()
    scalar = processor.accumulator[ns.scalar_nominal_key("ptz")]
    eft = processor.accumulator[ns.eft_nominal_key("ptz")]
    coeffs = np.asarray([[2., 1., 3., 4., 5., 6.], [2., 7., 8., 9., 10., 11.]])
    for systematic in ("nominal", "JESUp"):
        _fill_1d(processor, scalar, process="ttZ", systematic=systematic, is_data=False)
        _fill_1d(processor, eft, process="ttH", systematic=systematic, is_data=False, coeffs=coeffs)
    _fill_1d(processor, scalar, process="data", systematic="nominal", is_data=True)

    names = list(scalar.categorical_axes.name)
    # MC: 2**2 + 3**2; Data (same weights here): recorded as well.
    assert _by_key(ns.get_nominal_sumw2(scalar), names) == {"ttZ/nominal": 13.0, "data/nominal": 13.0}
    # EFT: SM point only, (2*2)**2 + (-3*2)**2.
    assert _by_key(ns.get_nominal_sumw2(eft), names) == {"ttH/nominal": 52.0}
    # JESUp storage exists but holds no recorded moment.
    raw = {tuple(k)[names.index("systematic")]: float(np.sum(v)) for k, v in scalar.nominal_sumw2().items()}
    assert raw["JESUp"] == 0.0
    # Yields are unaffected by the moment gate.
    yields = {tuple(k): v[..., 0].sum() for k, v in scalar.yield_coefficients().items()}
    assert all(value == pytest.approx(-1.0) for value in yields.values())


def test_fill_preserves_raw_count_classification():
    processor = _processor()
    scalar = processor.accumulator[ns.scalar_nominal_key("ptz")]
    _fill_1d(processor, scalar, process="ttZ", systematic="nominal", is_data=False)
    _fill_1d(processor, scalar, process="ttZ", systematic="JESUp", is_data=False)
    _fill_1d(processor, scalar, process="data", systematic="nominal", is_data=True)
    counts = scalar.raw_counts(flow=True)
    names = list(scalar.categorical_axes.name)
    recorded = {
        (tuple(k)[names.index("process")], tuple(k)[names.index("systematic")]): int(np.sum(v))
        for k, v in counts.items() if v is not None
    }
    # Only nominal MC fills are raw-count recorded; sumw2 does not change that.
    assert recorded.get(("ttZ", "nominal")) == 2
    assert recorded.get(("ttZ", "JESUp"), 0) == 0
    assert recorded.get(("data", "nominal"), 0) == 0
    restored = pickle.loads(pickle.dumps(scalar))
    assert restored.store_sumw2 and restored.track_raw_counts
    rebinned = rebin_histogram(scalar, [0, 100, 600])
    assert rebinned.store_sumw2 and rebinned.track_raw_counts
    np.testing.assert_allclose(
        sum(np.sum(v) for v in ns.get_nominal_sumw2(rebinned, flow=True).values()),
        sum(np.sum(v) for v in ns.get_nominal_sumw2(scalar, flow=True).values()),
    )


def test_two_dimensional_non_nominal_variance_stays_unrecorded():
    processor = _processor()
    two_d = processor.accumulator["lepton_pt_vs_eta"]
    pt_axis, eta_axis = two_d.dense_axes.name
    for systematic, weight in (("nominal", 2.0), ("JESUp", 3.0)):
        processor._fill_nominal_histogram(
            two_d,
            {pt_axis: np.asarray([30.0]), eta_axis: np.asarray([0.5]),
             "weight": np.asarray([weight]), "process": "ttZ", "channel": "3l",
             "appl": "isSR", "systematic": systematic},
            is_data=False, wgt_fluct=systematic,
        )
    by_syst = {
        tuple(k)[2]: (float(np.asarray(v.value).sum()), float(np.asarray(v.variance).sum()))
        for k, v in two_d.view(as_dict=True).items()
    }
    assert by_syst["nominal"] == (2.0, 4.0)
    # Yield is filled; the second moment is explicitly zero (not recorded).
    assert by_syst["JESUp"] == (3.0, 0.0)
    assert {tuple(k)[2] for k in ns.get_nominal_sumw2(two_d)} == {"nominal"}


def test_none_storage_records_no_moment():
    processor = _processor(fill_sumw2_hist=False)
    scalar = processor.accumulator[ns.scalar_nominal_key("ptz")]
    _fill_1d(processor, scalar, process="ttZ", systematic="nominal", is_data=False)
    with pytest.raises(ns.Sumw2NotStoredError):
        ns.get_nominal_sumw2(scalar)
    two_d = processor.accumulator["lepton_pt_vs_eta"]
    pt_axis, eta_axis = two_d.dense_axes.name
    processor._fill_nominal_histogram(
        two_d,
        {pt_axis: np.asarray([30.0]), eta_axis: np.asarray([0.5]),
         "weight": np.asarray([2.0]), "process": "ttZ", "channel": "3l",
         "appl": "isSR", "systematic": "JESUp"},
        is_data=False, wgt_fluct="JESUp",
    )
    (dense,) = two_d._dense_hists.values()
    assert dense.storage_type is hist.storage.Double
