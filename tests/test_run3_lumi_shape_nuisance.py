from __future__ import annotations

import json

import hist
import numpy as np
import pytest
import uproot

from topcoffea.modules.histEFT import HistEFT
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules.datacard_tools import (
    DatacardMaker,
    add_run3_luminosity_shape_nuisances,
    run3_luminosity_shape_factors,
    run3_luminosity_shape_factor,
)
from topeft.modules.paths import topeft_path


run3_years = ("2022", "2022EE", "2023", "2023BPix")


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


def select_merged_systematic(histogram, systematic):
    grouped = histogram.group(
        "process", {"sample": list(histogram.axes["process"])}
    )
    return grouped[
        {
            "process": hist.loc("sample"),
            "channel": hist.loc("3l_onZ_1b_3j"),
            "systematic": hist.loc(systematic),
        }
    ]


def physical_x_bin_index(histogram):
    x_axis = histogram.axes["x"]
    return x_axis.index(0.5) + int(x_axis.traits.underflow)


def merged_value(histogram, systematic, *, wc_value=None):
    selected = select_merged_systematic(histogram, systematic)
    if wc_value is None:
        values = selected.view(as_dict=True, flow=True)
    else:
        values = selected.eval({"ctG": wc_value})
    assert set(values) == {()}
    return float(values[()][physical_x_bin_index(selected)])


def merged_eft_coefficients(histogram, systematic):
    selected = select_merged_systematic(histogram, systematic)
    values = selected.view(as_dict=True, flow=True)
    assert set(values) == {()}
    coefficient_axis = selected.axes["quadratic_term"]
    coefficient_offset = int(coefficient_axis.traits.underflow)
    coefficient_indices = tuple(
        coefficient_axis.index(slot) + coefficient_offset
        for slot in range(len(coefficient_axis))
    )
    return np.asarray(
        values[()][physical_x_bin_index(selected), coefficient_indices]
    )


def root_physical_bin_value(root_histogram, physical_value):
    edges = root_histogram.axis().edges()
    bin_index = np.searchsorted(edges, physical_value, side="right") - 1
    assert 0 <= bin_index < len(edges) - 1
    return root_histogram.values()[bin_index]


def test_exact_annual_anchor_arithmetic_and_merged_response():
    annual_values = {
        "sample_2022": 2.0,
        "sample_2022EE": 3.0,
        "sample_2023": 5.0,
        "sample_2023BPix": 7.0,
    }
    histogram = add_run3_luminosity_shape_nuisances(
        make_scalar_hist(annual_values)
    )

    assert annual_values["sample_2022"] != annual_values["sample_2022EE"]
    assert annual_values["sample_2023"] != annual_values["sample_2023BPix"]
    for nuisance_name, year_factors in run3_luminosity_shape_factors.items():
        expected_up = sum(
            annual_values[f"sample_{year}"] * year_factors[year]
            for year in run3_years
        )
        expected_down = sum(
            annual_values[f"sample_{year}"] / year_factors[year]
            for year in run3_years
        )
        assert merged_value(histogram, f"{nuisance_name}Up") == pytest.approx(
            expected_up
        )
        assert merged_value(histogram, f"{nuisance_name}Down") == pytest.approx(
            expected_down
        )

    assert all(
        factors["2022"] == factors["2022EE"]
        and factors["2023"] == factors["2023BPix"]
        for factors in run3_luminosity_shape_factors.values()
    )
    assert run3_luminosity_shape_factor(
        "lumi_2", "2023BPix", "Down"
    ) == pytest.approx(1.0 / 1.0127)


def test_unity_years_contribute_nominal_content():
    histogram = add_run3_luminosity_shape_nuisances(
        make_scalar_hist({"sample_2022": 2.0, "sample_2023": 7.0})
    )
    assert set(histogram.axes["systematic"]) == {
        "nominal",
        "lumi_1Up",
        "lumi_1Down",
        "lumi_2Up",
        "lumi_2Down",
    }
    assert merged_value(histogram, "lumi_2Up") == pytest.approx(
        2.0 + 7.0 * 1.0127
    )
    assert merged_value(histogram, "lumi_2Down") == pytest.approx(
        2.0 + 7.0 / 1.0127
    )


@pytest.mark.parametrize(
    "process",
    (
        "data_obs2022",
        "charge_flips_2022",
        "fakes2022",
        "nonprompt2022",
    ),
)
def test_data_and_data_driven_processes_are_excluded(process):
    histogram = add_run3_luminosity_shape_nuisances(
        make_scalar_hist({process: 4.0})
    )
    assert set(histogram.axes["systematic"]) == {"nominal"}


@pytest.mark.parametrize(
    ("process", "expected_systematics"),
    (
        (
            "Diboson2022",
            {"nominal", "lumi_1Up", "lumi_1Down"},
        ),
        (
            "Diboson2023",
            {
                "nominal",
                "lumi_1Up",
                "lumi_1Down",
                "lumi_2Up",
                "lumi_2Down",
            },
        ),
    ),
)
def test_prompt_scalar_mc_omits_unity_only_coordinates(
    process,
    expected_systematics,
):
    histogram = make_scalar_hist({process: 4.0})
    add_run3_luminosity_shape_nuisances(histogram)
    assert set(histogram.axes["systematic"]) == expected_systematics


def test_eft_signal_scales_constant_linear_and_quadratic_coefficients():
    annual_values = {
        "ttll_2022": 2.0,
        "ttll_2022EE": 3.0,
        "ttll_2023": 5.0,
        "ttll_2023BPix": 7.0,
    }
    histogram = add_run3_luminosity_shape_nuisances(
        make_eft_hist(annual_values)
    )
    for nuisance_name, year_factors in run3_luminosity_shape_factors.items():
        expected_constant = sum(
            annual_values[f"ttll_{year}"] * year_factors[year]
            for year in run3_years
        )
        np.testing.assert_allclose(
            merged_eft_coefficients(
                histogram,
                f"{nuisance_name}Up",
            ),
            expected_constant * np.array([1.0, 2.0, 3.0]),
        )


@pytest.mark.parametrize("wc_value", (0.0, 0.4))
def test_eft_signal_consumer_smoke_at_sm_and_nonzero_wc(wc_value):
    annual_values = {"ttll_2022": 2.0, "ttll_2023": 7.0}
    histogram = add_run3_luminosity_shape_nuisances(
        make_eft_hist(annual_values)
    )
    polynomial_factor = 1.0 + 2.0 * wc_value + 3.0 * wc_value * wc_value
    expected = polynomial_factor * (
        annual_values["ttll_2022"] * 1.0138
        + annual_values["ttll_2023"] * 1.0017
    )
    assert merged_value(
        histogram,
        "lumi_1Up",
        wc_value=wc_value,
    ) == pytest.approx(expected)


@pytest.mark.parametrize("histogram_factory", (make_scalar_hist, make_eft_hist))
def test_tracked_raw_counts_mark_lumi_transients_unrecorded(histogram_factory):
    histogram = histogram_factory(
        {"Diboson2022": 2.0, "Diboson2023": 7.0},
        track_raw_counts=True,
    )
    nominal_raw_before = {
        key: value.copy()
        for key, value in histogram.raw_counts(flow=True).items()
    }

    maker = DatacardMaker.__new__(DatacardMaker)
    maker.do_nuisance = True
    maker.use_run3_systs = True
    maker.verbose = False
    merged = maker.correlate_years(histogram)

    for key, value in nominal_raw_before.items():
        np.testing.assert_array_equal(histogram.raw_counts(flow=True)[key], value)
    assert all(
        state is None
        for index, state in histogram._validated_raw_count_states().items()
        if histogram.index_to_categories(index).systematic != "nominal"
    )
    assert all(
        key.systematic == "nominal" for key in merged.raw_counts(flow=True)
    )
    merged_nominal = next(iter(merged.raw_counts(flow=True).values()))
    np.testing.assert_array_equal(merged_nominal, [0, 2, 0])
    merged._validated_raw_count_states()


def test_non_luminosity_shape_survives_year_merge():
    histogram = make_scalar_hist({"Diboson2022": 2.0, "Diboson2023": 7.0})
    for process, value in (("Diboson2022", 2.2), ("Diboson2023", 7.7)):
        histogram.fill(
            process=process,
            channel="3l_onZ_1b_3j",
            systematic="test_shapeUp",
            x=0.5,
            weight=value,
        )

    maker = DatacardMaker.__new__(DatacardMaker)
    maker.do_nuisance = True
    maker.use_run3_systs = True
    maker.verbose = False
    merged = maker.correlate_years(histogram)

    assert merged_value(merged, "nominal") == pytest.approx(9.0)
    assert merged_value(merged, "test_shapeUp") == pytest.approx(9.9)


def test_datacard_writer_emits_run3_shapes_and_no_global_lumi(tmp_path):
    annual_values = {
        "ttll_2022": 2.0,
        "ttll_2022EE": 3.0,
        "ttll_2023": 5.0,
        "ttll_2023BPix": 7.0,
    }
    histogram = make_eft_hist(annual_values, track_raw_counts=True)
    nominal_raw_before = {
        key: value.copy()
        for key, value in histogram.raw_counts(flow=True).items()
    }
    maker = DatacardMaker(
        hists={"x": histogram},
        out_dir=str(tmp_path),
        var_lst=["x"],
        year_lst=list(run3_years),
        do_nuisance=True,
        skip_missing_parton_rate_syst=True,
        binning_mode="processing",
        verbose=False,
    )

    maker.analyze(
        "x",
        "3l_onZ_1b_3j",
        {"ttll": ["ctG"]},
        crop_negative_bins=True,
        wcs_dict={"ctG": 0.4},
    )

    np.testing.assert_array_equal(
        next(iter(maker.hists["x"].raw_counts(flow=True).values())),
        sum(nominal_raw_before.values()),
    )
    root_path = tmp_path / "ttx_multileptons-3l_onZ_1b_3j_x.root"
    card_path = tmp_path / "ttx_multileptons-3l_onZ_1b_3j_x.txt"
    assert root_path.is_file()
    assert card_path.is_file()

    card = card_path.read_text()
    card_rows = [line.split() for line in card.splitlines()]
    shape_names = {
        row[0] for row in card_rows if len(row) > 1 and row[1] == "shape"
    }
    assert set(run3_luminosity_shape_factors) == {
        name for name in shape_names if name.startswith("lumi_")
    }
    assert "lumi_run3" not in card
    assert "pdf_scale_gg" in card

    with uproot.open(root_path) as root_file:
        root_keys = {key.split(";", 1)[0] for key in root_file.keys()}
        for nuisance_name, year_factors in run3_luminosity_shape_factors.items():
            for direction in ("Up", "Down"):
                assert f"ttll_sm_{nuisance_name}{direction}" in root_keys
            expected_up = sum(
                annual_values[f"ttll_{year}"] * year_factors[year]
                for year in run3_years
            )
            expected_down = sum(
                annual_values[f"ttll_{year}"] / year_factors[year]
                for year in run3_years
            )
            assert root_physical_bin_value(
                root_file[f"ttll_sm_{nuisance_name}Up"],
                0.5,
            ) == pytest.approx(expected_up)
            assert root_physical_bin_value(
                root_file[f"ttll_sm_{nuisance_name}Down"],
                0.5,
            ) == pytest.approx(
                expected_down,
            )


def test_run3_rate_config_has_no_global_lumi_or_new_lumi_topology():
    with open(topeft_path("params/rate_systs_run3.json")) as config_file:
        config = json.load(config_file)
    rate_uncertainties = config["rate_uncertainties"]
    assert "lumi_run3" not in rate_uncertainties
    assert not set(rate_uncertainties).intersection(run3_luminosity_shape_factors)
    assert "charge_flips_run3" in rate_uncertainties
    assert "pdf_scale" in rate_uncertainties
    assert "qcd_scale" in rate_uncertainties
    assert set(run3_luminosity_shape_factors) == {"lumi_1", "lumi_2"}
    assert all(
        set(year_factors) == set(run3_years)
        for year_factors in run3_luminosity_shape_factors.values()
    )


def test_do_nuisance_false_keeps_nominal_year_merge():
    maker = DatacardMaker.__new__(DatacardMaker)
    maker.do_nuisance = False
    histogram = maker.correlate_years(
        make_scalar_hist({"Diboson2022": 2.0, "Diboson2023": 7.0})
    )
    assert set(histogram.axes["systematic"]) == {"nominal"}
    assert merged_value(histogram, "nominal") == pytest.approx(9.0)


def test_existing_persisted_run3_lumi_template_is_rejected():
    histogram = make_scalar_hist({"Diboson2022": 4.0})
    histogram.fill(
        process="Diboson2022",
        channel="3l_onZ_1b_3j",
        systematic="lumi_1Up",
        x=0.5,
        weight=4.0,
    )
    with pytest.raises(RuntimeError, match="producer-side persistence"):
        add_run3_luminosity_shape_nuisances(histogram)
