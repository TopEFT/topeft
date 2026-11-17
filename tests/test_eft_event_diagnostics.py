"""Algebra, native fill semantics and observation-boundary regression tests."""

import csv
import io
import json
import math
import subprocess
import sys
from types import SimpleNamespace

import awkward as ak
import hist
import numpy as np
import pytest

from analysis.topeft_run2 import eft_event_diagnostics as diag
from topcoffea.modules import corrections, eft_helper
from topcoffea.modules.histEFT import HistEFT


def combined_fixture(channel, family, scalar_process, eft_process, scalar, constant, quadratic):
    from topcoffea.modules.sparseHist import SparseHist
    from topeft.modules.axis_binning import resolve_axis_edges
    edges = resolve_axis_edges(family, mode='fitting', channel=channel)
    axes = [hist.axis.StrCategory([], growth=True, name=n)
            for n in ('process', 'channel', 'systematic')]
    scalar_hist = SparseHist(*axes, hist.axis.Variable(edges, name=family))
    scalar_hist.fill(process=scalar_process, channel=channel, systematic='nominal',
                     **{family: np.array([edges[0] + 1]), 'weight': np.array([scalar])})
    eft_hist = HistEFT(*axes, hist.axis.Variable(edges, name=family), wc_names=['ctp'])
    eft_hist.fill(process=eft_process, channel=channel, systematic='nominal',
                  **{family: np.array([edges[0] + 1]), 'weight': np.array([1.]),
                     'eft_coeff': np.array([[constant, 0., quadratic]])})
    return {family+'__scalar_nominal': scalar_hist, family+'__eft_nominal': eft_hist}


@pytest.mark.parametrize('scalar,constant', [(0.06582480208393333, 0.22830001192466315),
                                            (3., 2.)])
def test_combined_run3_ttll_preserves_ownership_counterexamples(scalar, constant):
    channel = '2lss_4t_m_4j'
    components = combined_fixture(channel, 'lj0pt', 'TTto2L2Nu_central2022',
                                  'ttll_private2022', scalar, constant, -1.)
    bundle = diag.combined_producer_views(components, 'lj0pt', ['2022'], [channel])
    values = diag.combined_bin_coefficients(bundle, channel, 'ttll')
    assert values['combined'][0, 0] == pytest.approx(constant)
    assert values['scalar'][0, 0] == 0
    assert values['eft'][0, 0] == pytest.approx(constant)
    assert values['combined'][0, 0] != pytest.approx(scalar + constant)
    period = diag.combined_bin_coefficients(bundle, channel, 'ttll', period='2022')
    np.testing.assert_array_equal(period['combined'], values['combined'])


def test_combined_run2_block5_additive_route():
    channel = '3l_fwd_p_2j'
    components = combined_fixture(channel, 'lt', 'TTTo2L2Nu_centralUL18',
                                  'ttllJet_privateUL18', 3., 2., -1.)
    bundle = diag.combined_producer_views(components, 'lt', ['UL18'], [channel])
    values = diag.combined_bin_coefficients(bundle, channel, 'ttll')
    assert values['combined'][0, 0] == 5.
    assert values['scalar'][0, 0] == 3.
    assert values['eft'][0, 0] == 2.


def test_combined_ch145_period_ownership_route():
    channel = '2lss_fwd_p_6j'
    components = combined_fixture(channel, 'lt', 'TTto2L2Nu_central2022',
                                  'ttll_private2022', 3., 2., -1.)
    bundle = diag.combined_producer_views(components, 'lt', ['2022'], [channel])
    period = diag.combined_bin_coefficients(bundle, channel, 'ttll', period='2022')
    assert period['scalar'][0, 0] == 0
    assert period['eft'][0, 0] == 2.
    with pytest.raises(ValueError, match='absent'):
        diag.combined_bin_coefficients(bundle, channel, 'ttll', period='2023')


def test_combined_ch235_signed_denominator_is_preserved():
    channel = '3l_p_offZ_high_1b_5j'
    components = combined_fixture(channel, 'ptll', 'TTto2L2Nu_central2022',
                                  'tHq_private2022', 3., -2., 1.)
    bundle = diag.combined_producer_views(components, 'ptll', ['2022'], [channel])
    values = diag.combined_bin_coefficients(bundle, channel, 'tHq')
    assert values['combined'][0, 0] == -2.
    assert values['combined'][0, 2] == 1.
    # A sanitized package template must not redefine the signed denominator.
    assert values['combined'][0, 2] / values['combined'][0, 0] == -.5
    assert diag.package_bin_representation(-2., 0.) == 'SANITIZED_ZERO_REQUIRES_LINEAGE'
    assert diag.package_bin_representation(-2., None) == 'ABSENT_TEMPLATE_REQUIRES_LINEAGE'
    assert diag.package_bin_representation(2., 3.) == 'REPRESENTATION_MISMATCH'
    assert diag.package_bin_representation(2., 2.) == 'SIGNED_A0_REPRESENTED'


def test_mapping_follows_permuted_wc_names_and_native_remap():
    names = ["cpt", "cQq13", "ctG"]
    coeff = np.arange(20, dtype=float).reshape(2, 10)
    reordered = list(reversed(names))
    remapped = eft_helper.remap_coeffs(names, reordered, coeff)
    for wc in names:
        before, after = diag.diagonal_index(names, wc), diag.diagonal_index(reordered, wc)
        np.testing.assert_array_equal(coeff[:, before], remapped[:, after])
    with pytest.raises(ValueError):
        diag.diagonal_index(["c", "c"], "c")
    with pytest.raises(ValueError):
        diag.diagonal_index(names, "absent")


@pytest.mark.parametrize("weight_dtype", [np.float32, np.float64])
def test_products_match_native_histeft_edges_and_flow(weight_dtype):
    coeff = np.array([[1, -2, 3], [4, 5, -6], [7, 8, 9], [1, 2, 3], [2, 3, 4]], dtype=np.float32)
    weights = np.array([2, -3, 0, 1, -2], dtype=weight_dtype)
    coordinates = np.array([-1, 0, 50, 100, 101])
    edges = [0, 50, 100]
    histogram = HistEFT(hist.axis.StrCategory([], growth=True, name="channel"),
                        hist.axis.Variable(edges, name="x"), wc_names=["c"])
    histogram.fill(channel="test", x=coordinates, weight=weights, eft_coeff=coeff)
    observed = np.asarray(histogram[{"channel": "test"}].values(flow=True))[:, 1:-1]
    expected = np.zeros_like(observed)
    products = diag.event_products(coeff, weights)
    for index, source_bin in enumerate(np.searchsorted(edges, coordinates, side="right")):
        expected[source_bin] += products[index]
    np.testing.assert_array_equal(expected, observed)


def test_r6c1_frozen_events_and_mp009a3_sm_sumw2():
    # Durable R6C1 two-event fixture: source event 436 and event 282.
    coeff = np.array([[2.6525231078267097e-06, 0.015354156494140625],
                      [1.0978896170854568e-05, 0.016553878784179688]])
    weights = np.array([0.1475611033775082, -0.05174577494784256])
    products = diag.event_products(coeff, weights)
    np.testing.assert_allclose(products, [[3.914092365252464e-07, 0.0022656762737063236],
                                         [-5.681114904327709e-07, -0.0008565932860800277]], rtol=1e-14, atol=0)
    assert math.fsum(products[:, 0]) == pytest.approx(-1.767022539075245e-07, rel=1e-14, abs=0)
    assert math.fsum(products[:, 0] ** 2) == pytest.approx(4.759518559990206e-13, rel=1e-14, abs=0)
    assert math.fsum(products[:, 1]) == pytest.approx(0.0014090829876262957, rel=1e-14, abs=0)
    assert math.fsum(products[:, 1] ** 2) == pytest.approx(5.8670410349931515e-06, rel=1e-14, abs=0)
    # Squaring the scalar weight alone is not the SM statistical moment.
    assert not np.isclose(np.sum(weights**2), np.sum(products[:, 0]**2))


def test_r6c4_negative_loose_not_medium_and_native_zero_guard():
    rows = [{"subgroup": "bc", "category": "LOOSE_NOT_MEDIUM",
             "eM": 0.8696965801856155, "eL": 0.9495840951077332,
             "SFM": 1.1189123148223499, "SFL": 0.9981717698816024}]
    result = diag.btag_decomposition(rows)
    subgroup = result["subgroups"]["bc"]
    assert subgroup["jets"][0]["data_factor"] == pytest.approx(-0.025266176863462553, rel=1e-14)
    assert result["btagSF"] == pytest.approx(-0.3162719091724723, rel=1e-14)
    data, mc = corrections.get_method1a_wgt_doublewp(
        ak.Array([[rows[0]["eM"]]]), ak.Array([[rows[0]["eL"]]]),
        ak.Array([[rows[0]["SFM"]]]), ak.Array([[rows[0]["SFL"]]]),
        ak.Array([[False]]), ak.Array([[True]]), ak.Array([[False]]))
    # The diagnostic preserves the historical raw interval; the current helper
    # applies the accepted negative-data Loose-not-Medium fallback for this jet.
    assert float(data[0] / mc[0]) == pytest.approx(1.0, rel=1e-14)
    zero = dict(rows[0], eM=0.5, eL=0.5)
    assert diag.btag_decomposition([zero])["subgroups"]["bc"]["mc_after_guard"] == 1


def test_wide_selection_retains_native_masks_across_shards():
    selection = diag.wide_packed_selection()
    arrays = []
    for i in range(130):
        mask = np.arange(12) % (i % 5 + 2) == 0
        arrays.append(mask)
        selection.add(f"mask_{i}", mask)
    np.testing.assert_array_equal(selection.all("mask_0", "mask_66", "mask_129"), arrays[0] & arrays[66] & arrays[129])
    np.testing.assert_array_equal(selection.any("mask_0", "mask_66", "mask_129"), arrays[0] | arrays[66] | arrays[129])
    with pytest.raises(ValueError):
        selection.add("mask_0", arrays[0])


def test_capture_pairs_masks_native_bins_and_requested_terms():
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=diag.EVENT_FIELDS, delimiter="\t")
    writer.writeheader()
    context = {"context_id": "file_a", "group_id": "g", "local_root_path": "/local/a.root",
               "process": "tHq", "year_or_era": "2023"}
    owner = {"negative_row_id": "n1", "native_term_index": "2"}
    capture = diag.fill_capture(context, {("channel", "x"): (np.array([0, 50, 100]), {2: [owner]})},
                                {"channel": "isSR_3l"}, writer)
    capture.entry_start = 100
    values = {"ch_name": "channel", "dense_axis_name": "x", "syst_var": "nominal", "wgt_fluct": "nominal",
              "appl": "isSR_3l", "all_cuts_mask": ak.Array([False, True, True, True]),
              "combined_axis_mask": ak.Array([True, False, True]),
              "values_cut_map": {"x": ak.Array([10, 70])},
              "eft_coeffs_cut": np.array([[1, 0, 5], [2, 0, -3]], dtype=np.float32),
              "weights_flat": ak.Array([2.0, 4.0]),
              "events": SimpleNamespace(run=[1, 1, 1, 1], luminosityBlock=[2, 2, 2, 2], event=[10, 11, 12, 13])}
    capture(values)
    rows = list(csv.DictReader(io.StringIO(stream.getvalue()), delimiter="\t"))
    assert len(rows) == 1 and rows[0]["entry"] == "103"
    assert json.loads(rows[0]["qii"]) == [-12]
    assert rows[0]["event"] == "13" and rows[0]["source_bin"] == "2"
    assert capture.sums["n1"].value() == -12
    assert diag.mechanism(capture.signs["n1"], True) == "EFT_COEFFICIENT_ORIGIN"


def test_negative_ordinary_weights_serialize_actual_factor_product():
    assert diag.weight_product({"norm": 2, "btagSF": -0.5, "unit": 1}) == -1
    with pytest.raises(ValueError):
        diag.weight_product({})
    with pytest.raises(ValueError):
        diag.weight_product({"bad": math.nan})


def test_cancellation_and_incomplete_coverage_do_not_certify_zero():
    summed = diag.compensated_sum()
    for value in [1e16, 1, -1e16]:
        summed.add(value)
    assert summed.value() == 1
    assert diag.mechanism({"-1,1,-1": 2, "1,-1,-1": 3}, True) == "MIXED"
    assert diag.mechanism({"-1,1,-1": 2}, True) == "ORDINARY_WEIGHT_ORIGIN"
    assert diag.mechanism({"-1,-1,1": 2}, True) == "UNRESOLVED"
    assert diag.mechanism({"1,-1,-1": 2}, False) == "UNRESOLVED"


def test_observer_uses_current_processor_without_disk_mutation():
    original = diag.sha256(diag.PROCESSOR)
    module, instrumented_hash = diag.observed_processor(lambda _: None, original)
    assert diag.sha256(diag.PROCESSOR) == original
    assert instrumented_hash != original
    processor = module.AnalysisProcessor({}, [], hist_lst=["ptll"], fill_sumw2_hist=False,
                                         do_systematics=False, skip_control_regions=True, all_analysis=True)
    assert processor._fill_sumw2_hist is False
    with pytest.raises(ValueError, match="identity changed"):
        diag.observed_processor(lambda _: None, "wrong")


def test_invalid_event_factors_fail_before_serialization():
    with pytest.raises(ValueError):
        diag.event_products([[1, math.nan]], [1])
    with pytest.raises(ValueError):
        diag.event_products([[1, 2]], [1, 2])
    with pytest.raises(ValueError):
        diag.sign(math.inf)


def test_io_guard_denies_network_and_unlisted_or_writable_pickles():
    # Synthetic audit events exercise the guard without opening a pickle,
    # creating a socket, resolving a host or performing network I/O.
    script = '''
import sys
from analysis.topeft_run2.eft_event_diagnostics import install_io_guard
opened = install_io_guard(['/allowed_calibration.pkl.gz'])
sys.audit('open', '/allowed_calibration.pkl.gz', 'r', 0)
assert opened['/allowed_calibration.pkl.gz'] == 1
for event, args in [
    ('open', ('/analysis.pkl.gz', 'r', 0)),
    ('open', ('/allowed_calibration.pkl.gz', 'w', 0)),
    ('socket.connect', (None, ('127.0.0.1', 9))),
    ('socket.getaddrinfo', ('example.invalid', 443, 0, 0, 0)),
]:
    try:
        sys.audit(event, *args)
    except PermissionError:
        pass
    else:
        raise AssertionError(event)
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
