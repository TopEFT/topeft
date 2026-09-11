from __future__ import annotations

import hist
import numpy as np
import pytest

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.datacard_tools import (
    DatacardMaker,
    add_run2_luminosity_shape_nuisances,
    run2_luminosity_shape_factors,
    run2_luminosity_shape_factor,
)


run2_years = ("UL16APV", "UL16", "UL17", "UL18")


def make_scalar_hist(process_values, *, track_raw_counts=False):
    histogram = SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="x"),
        storage="Double",
        track_raw_counts=track_raw_counts,
    )
    for process, value in process_values.items():
        histogram.fill(
            process=process,
            channel="3l_onZ_1b_3j",
            systematic="nominal",
            x=0.5,
            weight=value,
            record_raw_count=True if track_raw_counts else None,
        )
    return histogram


def make_eft_hist(process_values, *, track_raw_counts=False):
    histogram = HistEFT(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="x"),
        wc_names=["ctG"],
        track_raw_counts=track_raw_counts,
    )
    for process, value in process_values.items():
        histogram.fill(
            process=process,
            channel="3l_onZ_1b_3j",
            systematic="nominal",
            x=np.array([0.5]),
            eft_coeff=np.array([[value, 2.0 * value, 3.0 * value]]),
            record_raw_count=True if track_raw_counts else None,
        )
    return histogram


def merged_value(histogram, systematic, *, wc_value=None):
    grouped = histogram.group(
        "process", {"sample": list(histogram.axes["process"])}
    )
    if wc_value is None:
        values = grouped.view(as_dict=True, flow=True)
    else:
        values = grouped.eval({"ctG": wc_value})
    matching = [
        value
        for key, value in values.items()
        if key.process == "sample"
        and key.channel == "3l_onZ_1b_3j"
        and key.systematic == systematic
    ]
    assert len(matching) == 1
    return float(matching[0][1])


def test_exact_annual_anchor_arithmetic_and_reciprocal_down():
    annual_values = {
        "sample_UL16APV": 2.0,
        "sample_UL16": 3.0,
        "sample_UL17": 5.0,
        "sample_UL18": 7.0,
    }
    histogram = add_run2_luminosity_shape_nuisances(
        make_scalar_hist(annual_values)
    )

    for nuisance_name, year_factors in run2_luminosity_shape_factors.items():
        expected_up = sum(
            annual_values[f"sample_{year}"] * year_factors[year]
            for year in run2_years
        )
        expected_down = sum(
            annual_values[f"sample_{year}"] / year_factors[year]
            for year in run2_years
        )
        assert merged_value(histogram, f"{nuisance_name}Up") == pytest.approx(
            expected_up
        )
        assert merged_value(histogram, f"{nuisance_name}Down") == pytest.approx(
            expected_down
        )

    assert all(
        factors["UL16APV"] == factors["UL16"]
        for factors in run2_luminosity_shape_factors.values()
    )
    assert run2_luminosity_shape_factor(
        "lumi_13TeV_1516_l", "UL16", "Down"
    ) == pytest.approx(1.0 / 1.0118)


def test_unaffected_years_contribute_nominal_content():
    histogram = add_run2_luminosity_shape_nuisances(
        make_scalar_hist({"sample_UL16": 2.0, "sample_UL18": 7.0})
    )
    assert merged_value(histogram, "lumi_13TeV_1516_lUp") == pytest.approx(
        2.0 * 1.0118 + 7.0
    )
    assert merged_value(histogram, "lumi_13TeV_1516_lDown") == pytest.approx(
        2.0 / 1.0118 + 7.0
    )


@pytest.mark.parametrize(
    "process",
    ("data_obsUL16", "charge_flips_UL16", "fakesUL16", "nonpromptUL16"),
)
def test_data_and_data_driven_processes_are_excluded(process):
    histogram = add_run2_luminosity_shape_nuisances(
        make_scalar_hist({process: 4.0})
    )
    assert set(histogram.axes["systematic"]) == {"nominal"}


def test_prompt_scalar_mc_is_included_from_nominal_only_input():
    histogram = make_scalar_hist({"DibosonUL16": 4.0})
    assert set(histogram.axes["systematic"]) == {"nominal"}
    add_run2_luminosity_shape_nuisances(histogram)
    expected_systematics = {"nominal"}
    expected_systematics.update(
        f"{name}{direction}"
        for name in run2_luminosity_shape_factors
        for direction in ("Up", "Down")
    )
    assert set(histogram.axes["systematic"]) == expected_systematics


def test_eft_signal_scales_all_coefficients_with_standard_shape_semantics():
    annual_values = {"ttll_UL16": 2.0, "ttll_UL18": 7.0}
    histogram = add_run2_luminosity_shape_nuisances(make_eft_hist(annual_values))
    wc_value = 0.4
    polynomial_factor = 1.0 + 2.0 * wc_value + 3.0 * wc_value * wc_value
    expected = polynomial_factor * (2.0 * 1.0118 + 7.0)
    assert merged_value(
        histogram, "lumi_13TeV_1516_lUp", wc_value=wc_value
    ) == pytest.approx(expected)


@pytest.mark.parametrize("histogram_factory", (make_scalar_hist, make_eft_hist))
def test_tracked_raw_counts_mark_lumi_transients_unrecorded(histogram_factory):
    histogram = histogram_factory(
        {"DibosonUL16": 2.0, "DibosonUL18": 7.0},
        track_raw_counts=True,
    )
    nominal_raw_before = {
        key: value.copy()
        for key, value in histogram.raw_counts(flow=True).items()
    }

    maker = DatacardMaker.__new__(DatacardMaker)
    maker.do_nuisance = True
    maker.use_run3_systs = False
    maker.verbose = False
    merged = maker.correlate_years(histogram)

    for key, value in nominal_raw_before.items():
        np.testing.assert_array_equal(
            histogram.raw_counts(flow=True)[key],
            value,
        )
    assert all(
        state is None
        for index, state in histogram._validated_raw_count_states().items()
        if histogram.index_to_categories(index).systematic != "nominal"
    )
    assert all(
        key.systematic == "nominal"
        for key in merged.raw_counts(flow=True)
    )
    merged_nominal = next(iter(merged.raw_counts(flow=True).values()))
    np.testing.assert_array_equal(merged_nominal, [0, 2, 0])
    merged._validated_raw_count_states()


def test_tracked_raw_counts_survive_copy_scale_and_additive_merge():
    histogram = make_scalar_hist(
        {"DibosonUL16": 2.0},
        track_raw_counts=True,
    )
    raw_before = next(iter(histogram.raw_counts(flow=True).values())).copy()

    scaled = histogram.copy()
    scaled.scale(3.0)
    np.testing.assert_array_equal(
        next(iter(scaled.raw_counts(flow=True).values())),
        raw_before,
    )

    merged = histogram.copy()
    merged += histogram.copy()
    np.testing.assert_array_equal(
        next(iter(merged.raw_counts(flow=True).values())),
        2 * raw_before,
    )


def test_datacard_writer_accepts_tracked_raw_count_input(tmp_path):
    histogram = make_eft_hist(
        {"DibosonUL16": 2.0, "DibosonUL18": 7.0},
        track_raw_counts=True,
    )
    nominal_raw_before = {
        key: value.copy()
        for key, value in histogram.raw_counts(flow=True).items()
    }
    maker = DatacardMaker(
        hists={"x": histogram},
        out_dir=str(tmp_path),
        var_lst=["x"],
        year_lst=["UL16", "UL18"],
        do_nuisance=True,
        skip_missing_parton_rate_syst=True,
        binning_mode="processing",
        verbose=False,
    )

    maker.analyze(
        "x",
        "3l_onZ_1b_3j",
        {"Diboson": []},
        crop_negative_bins=True,
        wcs_dict={},
    )

    np.testing.assert_array_equal(
        next(iter(maker.hists["x"].raw_counts(flow=True).values())),
        sum(nominal_raw_before.values()),
    )
    assert (tmp_path / "ttx_multileptons-3l_onZ_1b_3j_x.root").is_file()
    card = (tmp_path / "ttx_multileptons-3l_onZ_1b_3j_x.txt").read_text()
    card_rows = [line.split() for line in card.splitlines()]
    shape_names = {row[0] for row in card_rows if len(row) > 1 and row[1] == "shape"}
    assert set(run2_luminosity_shape_factors).issubset(shape_names)
    assert "lumi lnN" not in card


def test_run3_input_is_unchanged():
    histogram = make_scalar_hist({"Diboson2022": 4.0})
    before = histogram.view(as_dict=True, flow=True)
    before_values = {key: value.copy() for key, value in before.items()}
    add_run2_luminosity_shape_nuisances(histogram)
    after = histogram.view(as_dict=True, flow=True)
    assert set(histogram.axes["systematic"]) == {"nominal"}
    assert after.keys() == before_values.keys()
    for key in after:
        np.testing.assert_array_equal(after[key], before_values[key])


def test_run2_year_subset_omits_unity_only_coordinates():
    histogram = add_run2_luminosity_shape_nuisances(
        make_scalar_hist({"DibosonUL18": 4.0})
    )
    assert set(histogram.axes["systematic"]) == {
        "nominal",
        "lumi_13TeV_15161718_lUp",
        "lumi_13TeV_15161718_lDown",
    }


def test_do_nuisance_false_keeps_nominal_year_merge():
    maker = DatacardMaker.__new__(DatacardMaker)
    maker.do_nuisance = False
    histogram = maker.correlate_years(
        make_scalar_hist({"DibosonUL16": 2.0, "DibosonUL18": 7.0})
    )
    assert set(histogram.axes["systematic"]) == {"nominal"}
    assert merged_value(histogram, "nominal") == pytest.approx(9.0)


def test_obsolete_global_lumi_rate_is_not_loaded_for_run2():
    maker = DatacardMaker.__new__(DatacardMaker)
    maker.do_nuisance = True
    maker.use_run3_systs = False
    maker.skip_missing_parton_rate_syst = True
    maker.year_lst = list(run2_years)
    rate_systematics = maker.load_systematics("params/rate_systs_run2.json", None)
    assert "lumi" not in rate_systematics
    assert "charge_flips" in rate_systematics
    assert "pdf_scale_gg" in rate_systematics
    assert "qcd_scale_ttll" in rate_systematics


def test_existing_persisted_lumi_template_is_rejected():
    histogram = make_scalar_hist({"DibosonUL16": 4.0})
    histogram.fill(
        process="DibosonUL16",
        channel="3l_onZ_1b_3j",
        systematic="lumi_13TeV_1516_lUp",
        x=0.5,
        weight=4.0,
    )
    with pytest.raises(RuntimeError, match="producer-side persistence"):
        add_run2_luminosity_shape_nuisances(histogram)
