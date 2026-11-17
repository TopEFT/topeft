"""Raw EFT grey boundaries and full-context origin transport regressions."""
import copy
import io

import hist
import numpy as np
import pytest

from topcoffea.modules.histEFT import HistEFT
from topeft.modules import datacard_tools as dt
from topeft.modules.datacard_tools import (
    DatacardMaker, eft_grey_mask, suppress_eft_grey,
)
from topeft.modules.nominal_schema import materialize_legacy_histogram_dict
from tests.test_eft_event_diagnostics import combined_fixture
from analysis.topeft_run2 import eft_event_diagnostics as reference


@pytest.mark.parametrize("x,e0,a0,grey", [
    (5e-7, 1., 1., True), (5e-7, 1e-6, 1., False),
    (1e-6, 1., 1., False), (1e-8, 1e-6, 1., False),
    (1e-8, 1., 1e-6, False), (0., 0., 1., False),
    (0., 1., 0., False), (5e-7, np.inf, 1., False),
    (5e-7, 1., np.nan, False), (np.nan, 1., 1., False),
    (5e-7, 1e-320, 1., False),
])
@pytest.mark.parametrize("sign", [-1, 1])
def test_strict_grey_boundaries(x, e0, a0, grey, sign):
    assert bool(eft_grey_mask(sign*x, sign*e0, sign*a0)) is grey


def test_diagonal_and_mixed_are_independent_and_input_is_preserved():
    # Native order: constant, ci, ci^2, cj, ci*cj, cj^2.
    raw = np.array([[1., 4., 5e-7, 7., 2., 3.],
                    [1., 4., 2., 7., 5e-7, 3.]])
    before = raw.copy()
    output = suppress_eft_grey(raw, np.ones(2), ["ci", "cj"])
    np.testing.assert_array_equal(output[0], [1., 0., 0., 7., 2., 3.])
    np.testing.assert_array_equal(output[1], [1., 4., 2., 7., 0., 3.])
    np.testing.assert_array_equal(raw, before)
    np.testing.assert_array_equal(
        suppress_eft_grey(raw, np.full(2, 1e-6), ["ci", "cj"]), raw)
    with pytest.raises(ValueError, match="E0"):
        suppress_eft_grey(raw, None, ["ci", "cj"])


def make_producer(components, family, channel, year, **kwargs):
    return DatacardMaker(hists=copy.deepcopy(components), var_lst=[family],
                         channel_patterns=[channel], year_lst=[year],
                         do_nuisance=False, binning_mode="fitting", **kwargs)


@pytest.mark.parametrize("channel,family,scalar_process,eft_process,year,scalar,e0", [
    ("2lss_4t_m_4j", "lj0pt", "TTto2L2Nu_central2022", "ttll_private2022", "2022", 3., 2.),
    ("3l_fwd_p_2j", "lt", "TTTo2L2Nu_centralUL18", "ttllJet_privateUL18", "UL18", 3., 2.),
    ("2lss_fwd_p_6j", "lt", "TTto2L2Nu_central2022", "ttll_private2022", "2022", 3., 2.),
    ("3l_p_offZ_high_1b_5j", "ptll", "TTto2L2Nu_central2022", "tHq_private2022", "2022", 3., -2.),
])
def test_origin_transport_matches_qualified_combined_reference(
        channel, family, scalar_process, eft_process, year, scalar, e0):
    components = combined_fixture(channel, family, scalar_process, eft_process, scalar, e0, -1.)
    process = "tHq" if eft_process.startswith("tHq") else "ttll"
    bundle = reference.combined_producer_views(copy.deepcopy(components), family, [year], [channel])
    expected = reference.combined_bin_coefficients(bundle, channel, process)
    maker = make_producer(components, family, channel, year)
    actual = maker.nominal_origin_coefficients(family, channel, process)
    for origin in expected:
        np.testing.assert_allclose(actual[origin][1:], expected[origin], rtol=1e-10, atol=1e-10)
    assert "nominal_origin" not in maker.hists[family].axes.name
    with pytest.raises(ValueError, match="absent"):
        maker.nominal_origin_coefficients(family, channel, "missing_process")


@pytest.mark.parametrize("scalar,e0,grey", [(0., 1., True), (1.-1e-6, 1e-6, False)])
def test_canonical_scaling_path_uses_e0_before_serialization(scalar, e0, grey):
    channel, family = "3l_fwd_p_2j", "lt"
    components = combined_fixture(channel, family, "TTTo2L2Nu_centralUL18",
                                  "ttllJet_privateUL18", scalar, e0, 5e-7)
    maker = make_producer(components, family, channel, "UL18")
    origins = maker.nominal_origin_coefficients(family, channel, "ttll")
    assert origins["combined"][1, 0] == pytest.approx(1.)
    assert origins["eft"][1, 0] == pytest.approx(e0)
    ch_hist = maker.binning_view(maker.hists[family].integrate("channel", [channel]), family, channel)
    filtered = maker._scaling_histogram_with_grey(ch_hist, channel, "ttll", family)
    assert filtered.values(flow=True)[1, 3] == (0. if grey else 5e-7)
    assert origins["combined"][1, 2] == 5e-7
    assert filtered.make_scaling()[1, 2] == (0. if grey else 5e-7)


def test_uniform_input_does_not_infer_origin():
    channel, family = "3l_fwd_p_2j", "lt"
    split = combined_fixture(channel, family, "TTTo2L2Nu_centralUL18", "ttllJet_privateUL18", 3., 2., 1.)
    maker = make_producer(materialize_legacy_histogram_dict(split), family, channel, "UL18")
    with pytest.raises(ValueError, match="origin unavailable"):
        maker.nominal_origin_coefficients(family, channel, "ttll")


def test_pure_eft_absent_scalar_keeps_combined_support():
    channel, family = "2lss_4t_m_4j", "lj0pt"
    components = combined_fixture(channel, family, "TTto2L2Nu_central2022", "ttll_private2022", 0., 2., 1.)
    del components[family + "__scalar_nominal"]
    maker = make_producer(components, family, channel, "2022")
    arrays = maker.nominal_origin_coefficients(family, channel, "ttll")
    np.testing.assert_array_equal(arrays["scalar"], 0.)
    np.testing.assert_array_equal(arrays["eft"], arrays["combined"])


def final_cell_maker(a0, e0, terms=(4., 2., 7., 2., 3.), cell=1):
    """Small final HistEFT fixture; coefficient flow slots remain native."""
    channel, family = "3l_fwd_p_2j", "lt"
    axes = [hist.axis.StrCategory([], growth=True, name=name)
            for name in ("systematic", "channel", "process")]
    source = HistEFT(*axes, hist.axis.Variable([0., 1., 2.], name=family),
                     wc_names=["ci", "cj"])
    for systematic in ("nominal", "testUp", "testDown"):
        source.fill(process="ttll", channel=channel, systematic=systematic,
                    lt=np.array([.5]), eft_coeff=np.array([[1., 0., 0., 0., 0., 0.]]))
    for key, values in source.view(flow=True).items():
        values[...] = 0
        values[:, 1] = 1.
        values[cell, 1:-1] = ([a0, *terms] if key.systematic == "nominal"
                                else [3., 2., 1., 4., 5., 6.])
    scalar, eft = copy.deepcopy(source), copy.deepcopy(source)
    for values in scalar.view(flow=True).values():
        values[...] = 0
        values[cell, 1] = a0 - e0
    for key, values in eft.view(flow=True).items():
        values[...] -= scalar.view(flow=True)[key]
    maker = object.__new__(DatacardMaker)
    maker.hists = {family: source}
    maker.nominal_origins = {family: {"scalar": scalar, "eft": eft}}
    maker.binning_mode = "processing"
    maker.verbose = False
    maker.coeffs = []
    maker.tolerance = 1e-4
    maker.use_AAC = False
    maker.wc_ranges = {name: (-1., 1.) for name in source.wc_names}
    return maker, channel, family


@pytest.mark.parametrize("a0,e0,terms,cell", [
    (-1e-3, 1., (4., 2., 7., 2., 3.), 1),  # C1
    (-1e-7, 1., (4., 2., 7., 2., 3.), 1),  # C2
    (0., 0., (0., 0., 0., 0., 0.), 1),     # C3
    (0., 1., (4., 2., 7., 2., 3.), 1),     # C4
    (1e-7, 1., (4., 2., 7., 2., 3.), 1),   # C5
    (.999e-5, 1., (4., 2., 7., 2., 3.), 1), # C6
    (1e-5, 1e-5, (4., 2., 7., 2., 3.), 1), # C7
    (2e-5, 2e-5, (4., 2., 7., 2., 3.), 1), # C8
    (1e-8, 2., (4., 2., 7., 2., 3.), 1),   # C9 signed cancellation
    (1., 1., (4., 5e-7, 7., 2., 3.), 1),   # C10 diagonal
    (1., 1., (4., 2., 7., 5e-7, 3.), 1),   # C11 mixed
    (1., 0., (4., 5e-7, 7., 5e-7, 3.), 1), # C12 E0=0
    (1e-7, 1., (4., 2., 7., 2., 3.), 3),   # C13 source overflow
])
def test_final_cell_policy_model_templates_and_serialized_multiplier(a0, e0, terms, cell, monkeypatch):
    maker, channel, family = final_cell_maker(a0, e0, terms, cell)
    source = maker.hists[family]
    before = {key: values.copy() for key, values in source.view(flow=True).items()}
    ch_hist = source.integrate("channel", [channel])
    raw_hist = maker._scaling_histogram_for_json(ch_hist, channel, "ttll")
    raw = np.asarray(raw_hist.values(flow=True))[..., 1:-1]
    mask = dt.eft_null_cell_mask(raw[:, 0])
    np.testing.assert_array_equal(mask, raw[:, 0] < 1e-5)
    original_grey = dt.suppress_eft_grey
    def grey_for_survivors(coefficients, origin, names):
        assert np.all(coefficients[:, 0] >= 1e-5)
        return original_grey(coefficients, origin, names)
    monkeypatch.setattr(dt, "suppress_eft_grey", grey_for_survivors)
    model = maker._scaling_histogram_with_grey(ch_hist, channel, "ttll", family, mask)
    model_coeffs = np.asarray(model.values(flow=True))[..., 1:-1]
    expected = raw.copy()
    if a0 < 1e-5:
        expected[cell] = 0
    elif e0 != 0:
        # Independent scalar oracle, retaining raw X and strict ratios.
        for diagonal, linear in ((2, 1), (5, 3)):
            x = raw[cell, diagonal]
            if abs(x) < 1e-6 and abs(x / e0) < .01 and abs(x / a0) < .01:
                expected[cell, [diagonal, linear]] = 0
        x = raw[cell, 4]
        if abs(x) < 1e-6 and abs(x / e0) < .01 and abs(x / a0) < .01:
            expected[cell, 4] = 0
    np.testing.assert_array_equal(model_coeffs, expected)
    # Native make_scaling rejects zero+BSM and skips zero-only rows. Raising
    # floating errors plus the exact pre-normalization postimage catch division.
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        normalized = model.make_scaling()
    np.testing.assert_array_equal(normalized[mask], 0.)
    payload, = maker.make_scalings_json([], channel, family, "ttll", ["ci", "cj"], normalized, mask)
    serialized = np.asarray(payload["scaling"])
    expected_scaled = expected.copy()
    expected_scaled[:, [1, 3, 4]] /= 2  # actual interference off-diagonal packing
    expected_scaled[~mask] /= raw[~mask, 0, None]
    expected_scaled[mask, 0] = 1
    np.testing.assert_array_equal(serialized, expected_scaled[1:])
    assert np.isfinite(serialized).all()
    proc = maker.select_final_sr_appl(ch_hist.integrate("process", ["ttll"]), channel, "ttll")
    templates = maker.decompose(proc, None, [])['sm']
    for arr in templates.values():
        arr[1][...] = 9.
    for crop in (False, True):
        sanitized = dt._sanitize_negative_template_bins(templates, mask, crop_negative_bins=crop)
        for key, arr in sanitized.items():
            np.testing.assert_array_equal(arr[0][mask], 0.)
            np.testing.assert_array_equal(arr[1][mask], 0.)
            np.testing.assert_array_equal(arr[0][~mask], templates[key][0][~mask])
            root_hist = dt.to_hist(arr, "fixture")
            np.testing.assert_array_equal(root_hist.values(), arr[0][1:])
            np.testing.assert_array_equal(root_hist.variances(), arr[1][1:])
    for key, values in source.view(flow=True).items():
        np.testing.assert_array_equal(values, before[key])


@pytest.mark.parametrize("a0", [np.nan, np.inf, -np.inf])
def test_nonfinite_a0_fails_each_dependent_realization(a0):
    with pytest.raises(ValueError, match="Nonfinite.*A0"):
        dt.eft_null_cell_mask([1., a0])
    maker, channel, family = final_cell_maker(1., 1.)
    # Inject after construction, since source validation may already reject it.
    for key, values in maker.hists[family].view(flow=True).items():
        if key.systematic == "nominal":
            values[1, 1] = a0
    with pytest.raises(ValueError):
        maker.get_selected_wcs(family)
    with pytest.raises(ValueError):
        maker._scaling_histogram_with_grey(
            maker.hists[family].integrate("channel", [channel]), channel, "ttll", family)


@pytest.mark.parametrize("cell", [1, 3])
@pytest.mark.parametrize("a0,quadratic,selected", [
    (1e-7, 2., False), (-1e-3, 2., False), (0., 2., False), # C15
    (1., .999e-4, False), (1., 1e-4, False), (1., 1.001e-4, True), # C16
    (1e-5, 1.001e-9, True),
])
def test_non_aac_selection_uses_surviving_final_cells(a0, quadratic, selected, cell):
    maker, _, family = final_cell_maker(a0, a0, (0., quadratic, 0., 0., 0.), cell)
    assert maker.get_selected_wcs(family)["ttll"] == ({"ci"} if selected else set())
    assert maker.tolerance == 1e-4


def test_selection_is_not_scaling_domain_and_aac_retains_historic_guard():
    maker, channel, family = final_cell_maker(1e-7, 1., (4., 2., 0., 0., 0.))
    assert maker.get_selected_wcs(family)["ttll"] == set()
    maker.use_AAC = True
    assert maker.get_selected_wcs(family)["ttll"] == {"ci"}
    ch = maker.hists[family].integrate("channel", [channel])
    aac = maker._scaling_histogram_with_grey(ch, channel, "ttll", family)
    assert np.asarray(aac.values(flow=True))[1, 1] == 1e-7
    maker.use_AAC = False
    # Below selection tolerance but not grey: its WC remains in the payload.
    maker.hists[family].view(flow=True)[next(k for k in maker.hists[family].categorical_keys
                                          if k.systematic == "nominal")][1, 1:-1] = [1., 0., 2e-6, 0., 0., 0.]
    maker.nominal_origins[family]["scalar"] = maker.hists[family] * 0
    maker.nominal_origins[family]["eft"] = copy.deepcopy(maker.hists[family])
    assert maker.get_selected_wcs(family)["ttll"] == set()
    ch = maker.hists[family].integrate("channel", [channel])
    model = maker._scaling_histogram_with_grey(ch, channel, "ttll", family)
    payload, = maker.make_scalings_json([], channel, family, "ttll", ["ci", "cj"], model.make_scaling(),
                                       dt.eft_null_cell_mask(np.asarray(model.values(flow=True))[:, 1]))
    assert len(payload["parameters"]) == 3
    assert payload["scaling"][0][2] == 2e-6


@pytest.mark.parametrize("wc_list", [["cj", "ci"], ["ci"], ["ci", "cj", "ck"]])
def test_null_serialization_respects_native_wc_remapping(wc_list):
    maker, channel, family = final_cell_maker(1e-7, 1.)
    maker.wc_ranges["ck"] = (-1., 1.)
    model = maker._scaling_histogram_with_grey(maker.hists[family].integrate("channel", [channel]), channel, "ttll", family)
    normalized = model.make_scaling(wc_list=wc_list)
    payload, = maker.make_scalings_json([], channel, family, "ttll", wc_list, normalized, np.array([False, True, False, False]))
    assert payload["scaling"][0] == [1.] + [0.] * (len(normalized[0]) - 1)


def test_non_aac_selection_projects_final_sr_without_ar_support():
    maker, _, family = final_cell_maker(1e-7, 1.)
    channel = "3l_p_offZ_none_1b_2j"
    sr = maker._get_missing_parton_channel_contract().expected_sr_appl(channel)
    source = maker.hists[family]
    with_appl = source.empty_from_axes(categorical_axes=[
        *(hist.axis.StrCategory([channel], growth=True, name="channel")
          if axis.name == "channel" else axis for axis in source.categorical_axes),
        hist.axis.StrCategory([sr, "test_ar"], growth=True, name="appl"),
    ])
    for key, values in source.view(flow=True).items():
        coordinates = key._asdict()
        coordinates["channel"] = channel
        base = tuple(coordinates[name] for name in source.categorical_axes.name)
        with_appl[(*base, sr)] = values
        with_appl[(*base, "test_ar")] = values * 1e8
    maker.hists[family] = with_appl
    assert maker.get_selected_wcs(family)["ttll"] == set()


@pytest.mark.parametrize("crop", [False, True])
def test_analyze_reuses_mask_for_templates_variations_asimov_and_scaling(monkeypatch, crop):
    # Run the maintained producer with memory-only ROOT/text writers.
    channel, family = "3l_fwd_p_2j", "lt"
    components = combined_fixture(channel, family, "TTTo2L2Nu_centralUL18", "ttllJet_privateUL18", 0., 1., 2.)
    maker = make_producer(components, family, channel, "UL18")
    fixture, _, _ = final_cell_maker(1e-7, 1.)
    maker.hists, maker.nominal_origins = fixture.hists, fixture.nominal_origins
    maker.binning_mode = "processing"
    maker.wc_ranges = fixture.wc_ranges
    maker.out_dir = "/memory_only_datacard_test"
    for hist_source in [maker.hists[family], *maker.nominal_origins[family].values()]:
        for key, values in hist_source.view(flow=True).items():
            values[-1] = values[1] # C13/C14 also at the last serialized bin
    written = {}
    class memory_root(dict):
        def __enter__(self): return self
        def __exit__(self, *args): return False
    monkeypatch.setattr(dt.uproot, "recreate", lambda path: memory_root_proxy)
    memory_root_proxy = memory_root()
    original_open = open
    def memory_open(path, *args, **kwargs):
        if str(path).startswith(maker.out_dir + "/"):
            assert args == ("w",)
            return io.StringIO()
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr("builtins.open", memory_open)
    maker.analyze(family, channel, {"ttll": []}, crop, {})
    written.update(memory_root_proxy)
    for name, histogram in written.items():
        np.testing.assert_array_equal(histogram.values()[[0, -1]], 0.)
    assert {"ttll_sm", "data_obs"}.issubset(written)
    assert any("test" in name for name in written)
    assert maker.scalings[0]["scaling"][0] == [1., 0., 0., 0., 0., 0.]
    assert maker.scalings[0]["scaling"][-1] == [1., 0., 0., 0., 0., 0.]
    # Raw provenance is retained by the real producer path.
    assert maker.nominal_origin_coefficients(family, channel, "ttll")["combined"][1, 0] == 1e-7
