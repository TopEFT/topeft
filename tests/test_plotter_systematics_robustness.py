import hist
import numpy as np
import pytest

from analysis.topeft_run2 import make_cr_and_sr_plots


def test_signed_display_allocation_preserves_hand_checked_total_and_inputs():
    grouped = np.array([[4.0, 2.0, 1.0], [2.0, -1.0, -5.0],
                        [-1.0, 3.0, 1.0]])
    original = grouped.copy()
    display, visible = make_cr_and_sr_plots._signed_display_stack(grouped)
    np.testing.assert_array_equal(grouped, original)
    np.testing.assert_allclose(visible, [5.0, 4.0, 0.0])
    np.testing.assert_allclose(display, [[10.0 / 3.0, 1.6, 0.0],
                                             [5.0 / 3.0, 0.0, 0.0],
                                             [0.0, 2.4, 0.0]])
    np.testing.assert_allclose(display.sum(axis=0), visible)


@pytest.fixture(autouse=True)
def _patch_eval_without_underflow(monkeypatch):
    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "_eval_without_underflow",
        lambda hist_slice: make_cr_and_sr_plots._values_without_flow(
            hist_slice,
            include_overflow=False,
        ),
    )


def _make_sparse_hist(systematic_weights):
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    channel_axis = hist.axis.StrCategory([], name="channel", growth=True)
    syst_axis = hist.axis.StrCategory([], name="systematic", growth=True)
    met_axis = hist.axis.Regular(1, 0.0, 1.0, name="met")

    hist_obj = make_cr_and_sr_plots.SparseHist(
        process_axis, channel_axis, syst_axis, met_axis
    )

    for systematic, weight in systematic_weights.items():
        hist_obj.fill(
            process="ttH_central2022",
            channel="2lss_em_CR_1j",
            systematic=systematic,
            met=0.5,
            weight=weight,
        )

    return hist_obj


def _make_process_sparse_hist(process_systematic_weights):
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    syst_axis = hist.axis.StrCategory([], name="systematic", growth=True)
    met_axis = hist.axis.Regular(1, 0.0, 1.0, name="met")

    hist_obj = make_cr_and_sr_plots.SparseHist(
        process_axis, syst_axis, met_axis
    )

    for process_name, systematic_weights in process_systematic_weights.items():
        for systematic, weight in systematic_weights.items():
            hist_obj.fill(
                process=process_name,
                systematic=systematic,
                met=0.5,
                weight=weight,
            )

    return hist_obj


def test_get_shape_syst_arrs_skips_orphan_and_keeps_valid_pairs():
    hist_obj = _make_sparse_hist(
        {
            "nominal": 10.0,
            "FooUp": 12.0,
            "BarUp": 14.0,
            "BarDown": 8.0,
        }
    )

    (shape_m, shape_p), details = make_cr_and_sr_plots.get_shape_syst_arrs(
        hist_obj,
        group_type="CR",
        return_details=True,
    )

    assert np.allclose(shape_p, np.array([16.0]))
    assert np.allclose(shape_m, np.array([4.0]))
    assert details["valid_bases"] == ("Bar",)
    assert details["evaluated_bases"] == ("Bar",)
    assert details["nonapplicable_bases"] == ()

    skipped = {
        entry["base"]: tuple(entry.get("missing", ()))
        for entry in details["skipped_orphans"]
    }
    assert "Foo" in skipped
    assert skipped["Foo"] == ("FooDown",)


@pytest.mark.parametrize(
    ("years", "expected"),
    [
        (("2016APV", "2016", "2017", "2018"), "run2"),
        (("UL16APV", "UL16", "UL17", "UL18"), "run2"),
        (("run2",), "run2"),
        (("2022", "2022EE", "2023", "2023BPix"), "run3"),
        (("run3",), "run3"),
    ],
)
def test_diboson_njets_era_resolution(years, expected):
    assert make_cr_and_sr_plots._resolve_diboson_njets_era(years) == expected


@pytest.mark.parametrize("years", [(), ("unknown",), ("UL18", "2022")])
def test_diboson_njets_era_rejects_unresolved_years(years):
    with pytest.raises(ValueError, match="Diboson njets total uncertainty"):
        make_cr_and_sr_plots._resolve_diboson_njets_era(years)


def test_diboson_njets_factors_use_the_resolved_era_and_preserve_nominal():
    nominal = np.arange(1.0, 10.0)
    expected_factors = {
        "run2": [1.0, 1.0, 1.2, 1.48, 1.77, 2.05, 2.34, 2.62, 2.62],
        "run3": [1.0, 1.0, 0.86, 1.43, 1.43, 1.72, 2.0, 2.29, 2.29],
    }
    before = nominal.copy()
    for era, factors in expected_factors.items():
        contribution = make_cr_and_sr_plots.get_diboson_njets_syst_arr(
            nominal, bin0_njets=0, era=era
        )
        np.testing.assert_allclose(contribution, (nominal * (np.array(factors) - 1)) ** 2)
    np.testing.assert_array_equal(nominal, before)
    assert make_cr_and_sr_plots._cached_get_jet_dependent_syst_dict("run2")["4"] == 1.77
    assert make_cr_and_sr_plots._cached_get_jet_dependent_syst_dict("run3")["4"] == 1.43


@pytest.mark.parametrize(
    ("rate_ok", "shape_ok", "details", "failure"),
    [
        (False, True, {}, "rate=missing rate"),
        (True, False, {}, "shape=missing shape"),
        (True, True, {"skipped_orphans": ({"base": "Foo"},)}, "Foo"),
        (True, True, {"skipped_failed": ({"base": "Bar"},)}, "Bar"),
    ],
)
def test_total_systematics_fail_closed(rate_ok, shape_ok, details, failure):
    with pytest.raises(ValueError, match="Incomplete total uncertainty") as exc_info:
        make_cr_and_sr_plots._require_complete_total_systematics(
            "SR", "2lss_p_4j", "njets", ("UL16",), rate_ok, shape_ok,
            details, rate_error="missing rate", shape_error="missing shape",
        )
    message = str(exc_info.value)
    assert "2lss_p_4j" in message
    assert "njets" in message
    assert "UL16" in message
    assert failure in message


def test_total_systematics_accepts_complete_pair_and_declared_no_shape():
    make_cr_and_sr_plots._require_complete_total_systematics(
        "SR", "2lss_p_4j", "njets", ("UL16",), True, True,
        {"valid_bases": ("Foo",), "skipped_orphans": (), "skipped_failed": ()},
    )
    make_cr_and_sr_plots._require_complete_total_systematics(
        "SR", "nominal_only", "met", ("UL16",), True, True,
        {"valid_bases": (), "skipped_orphans": (), "skipped_failed": ()},
    )


@pytest.mark.parametrize("variation", [np.array([1.0]), np.array([1.0, 2.0, 3.0])])
def test_shape_variation_length_mismatch_fails_closed(variation):
    with pytest.raises(ValueError, match="silent truncation or zero padding is forbidden"):
        make_cr_and_sr_plots._match_variation_length(np.array([1.0, 2.0]), variation)


def test_emit_systematics_summary_reports_nominal_only_rate_mode(capsys):
    hist_obj = _make_sparse_hist({"nominal": 5.0})

    (_, _), details = make_cr_and_sr_plots.get_shape_syst_arrs(
        hist_obj,
        group_type="CR",
        return_details=True,
    )

    make_cr_and_sr_plots._SYSTEMATICS_SUMMARY_EMITTED.clear()
    make_cr_and_sr_plots._emit_systematics_summary_once(
        "CR",
        ("lumi", "pdf_scale"),
        details,
        rate_calc_ok=True,
        shape_calc_ok=True,
    )

    out = capsys.readouterr().out
    assert "No shape systematics found on pkl axis." in out
    assert "Rate systematics from rate_systs.json: lumi, pdf_scale" in out
    assert "Rate systematic computation succeeded." in out
    assert "Shape systematic computation succeeded." in out
    assert "renormfact" not in out


def test_emit_systematics_summary_logs_orphan_pairs_once(capsys):
    hist_obj = _make_sparse_hist(
        {
            "nominal": 10.0,
            "FooUp": 12.0,
            "BarUp": 14.0,
            "BarDown": 8.0,
        }
    )

    (_, _), details = make_cr_and_sr_plots.get_shape_syst_arrs(
        hist_obj,
        group_type="CR",
        return_details=True,
    )

    make_cr_and_sr_plots._SYSTEMATICS_SUMMARY_EMITTED.clear()
    make_cr_and_sr_plots._emit_systematics_summary_once(
        "CR",
        ("lumi",),
        details,
        rate_calc_ok=True,
        shape_calc_ok=True,
    )
    make_cr_and_sr_plots._emit_systematics_summary_once(
        "CR",
        ("lumi",),
        details,
        rate_calc_ok=True,
        shape_calc_ok=True,
    )

    out = capsys.readouterr().out
    assert out.count("Systematics summary") == 1
    assert "Skipping shape systematic 'Foo'" in out
    assert "FooDown" in out


def test_emit_systematics_summary_reports_component_failures(capsys):
    make_cr_and_sr_plots._SYSTEMATICS_SUMMARY_EMITTED.clear()
    make_cr_and_sr_plots._emit_systematics_summary_once(
        "CR",
        ("lumi",),
        {"valid_bases": (), "skipped_orphans": (), "skipped_failed": ()},
        rate_calc_ok=False,
        shape_calc_ok=False,
    )

    out = capsys.readouterr().out
    assert "Shape systematic computation failed; shape uncertainties will be omitted." in out
    assert "Rate systematic computation failed; rate uncertainties will be omitted." in out
    assert "Shape systematic computation succeeded." not in out


def test_emit_systematics_summary_no_shape_does_not_imply_rate_usage_on_rate_failure(capsys):
    make_cr_and_sr_plots._SYSTEMATICS_SUMMARY_EMITTED.clear()
    make_cr_and_sr_plots._emit_systematics_summary_once(
        "CR",
        ("lumi",),
        {"valid_bases": (), "skipped_orphans": (), "skipped_failed": ()},
        rate_calc_ok=False,
        shape_calc_ok=True,
    )

    out = capsys.readouterr().out
    assert "No shape systematics found on pkl axis." in out
    assert "Rate systematic computation failed; rate uncertainties will be omitted." in out
    assert "using rate-only systematics" not in out


def test_emit_systematics_summary_mentions_renormfact_only_when_present(capsys):
    make_cr_and_sr_plots._SYSTEMATICS_SUMMARY_EMITTED.clear()
    make_cr_and_sr_plots._emit_systematics_summary_once(
        "CR",
        ("lumi",),
        {
            "valid_bases": (),
            "skipped_orphans": (),
            "skipped_failed": (),
            "renormfact_present": True,
        },
        rate_calc_ok=True,
        shape_calc_ok=True,
    )

    out = capsys.readouterr().out
    assert "renormfact' present on axis and explicitly skipped by design." in out


def test_filter_existing_processes_keeps_present_and_reports_absent():
    present, missing = make_cr_and_sr_plots.filter_existing_processes(
        ["ttH_central2022", "WJetsToLNu_centralUL17"],
        ["ttH_central2022", "ttbar_central2022"],
    )

    assert present == ["ttH_central2022"]
    assert missing == ["WJetsToLNu_centralUL17"]


def test_filter_existing_processes_requires_process_axis():
    hist_obj = hist.Hist(
        hist.axis.StrCategory(["nominal"], name="systematic"),
        hist.axis.Regular(1, 0.0, 1.0, name="met"),
    )

    with pytest.raises(KeyError, match="process"):
        make_cr_and_sr_plots._process_axis_labels(hist_obj)


def test_decorrelated_systematic_uses_present_subset_when_group_member_absent():
    hist_obj = _make_process_sparse_hist(
        {
            "ttH_central2022": {
                "nominal": 10.0,
                "renormUp": 13.0,
                "renormDown": 7.0,
            },
        }
    )

    p_arr, m_arr = make_cr_and_sr_plots.get_decorrelated_uncty(
        "renorm",
        {"Signal": ["ttH_central2022", "WJetsToLNu_centralUL17"]},
        ["ttH_central2022"],
        hist_obj,
        np.array([0.0]),
    )

    assert np.allclose(p_arr, np.array([3.0]))
    assert np.allclose(m_arr, np.array([-3.0]))


def test_decorrelated_systematic_all_absent_group_returns_zero_contribution():
    hist_obj = _make_process_sparse_hist(
        {
            "ttH_central2022": {
                "nominal": 10.0,
                "renormUp": 13.0,
                "renormDown": 7.0,
            },
        }
    )

    p_arr, m_arr = make_cr_and_sr_plots.get_decorrelated_uncty(
        "renorm",
        {"Singleboson": ["WJetsToLNu_centralUL17"]},
        ["ttH_central2022"],
        hist_obj,
        np.array([0.0]),
    )

    assert np.allclose(p_arr, np.array([0.0]))
    assert np.allclose(m_arr, np.array([-0.0]))


def test_renorm_systematic_is_not_skipped_when_group_map_has_missing_process(monkeypatch):
    hist_obj = _make_process_sparse_hist(
        {
            "ttH_central2022": {
                "nominal": 10.0,
                "renormUp": 13.0,
                "renormDown": 7.0,
                "factUp": 12.0,
                "factDown": 8.0,
            },
            "ttbar_central2022": {
                "nominal": 20.0,
                "renormUp": 24.0,
                "renormDown": 16.0,
                "factUp": 25.0,
                "factDown": 15.0,
            },
        }
    )

    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "CR_GRP_MAP",
        {
            "Signal": ["ttH_central2022"],
            "Top": ["ttbar_central2022", "WJetsToLNu_centralUL17"],
            "Singleboson": ["MissingSingleboson"],
        },
    )

    (shape_m, shape_p), details = make_cr_and_sr_plots.get_shape_syst_arrs(
        hist_obj,
        group_type="CR",
        return_details=True,
    )

    assert details["valid_bases"] == ("renorm", "fact")
    assert details["skipped_failed"] == ()
    assert np.allclose(shape_p, np.array([3.0**2 + 4.0**2 + 2.0**2 + 5.0**2]))
    assert np.allclose(shape_m, shape_p)


def test_renorm_all_present_grouping_behavior_is_unchanged(monkeypatch):
    hist_obj = _make_process_sparse_hist(
        {
            "ttbar_central2022": {
                "nominal": 20.0,
                "renormUp": 24.0,
                "renormDown": 16.0,
            },
            "TTTo2L2Nu_central2022": {
                "nominal": 5.0,
                "renormUp": 6.0,
                "renormDown": 4.0,
            },
        }
    )
    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "CR_GRP_MAP",
        {"Top": ["ttbar_central2022", "TTTo2L2Nu_central2022"]},
    )

    (shape_m, shape_p), details = make_cr_and_sr_plots.get_shape_syst_arrs(
        hist_obj,
        group_type="CR",
        return_details=True,
    )

    assert details["valid_bases"] == ("renorm",)
    assert details["skipped_failed"] == ()
    assert np.allclose(shape_p, np.array([5.0**2]))
    assert np.allclose(shape_m, np.array([5.0**2]))


def test_sr_ttbar_draw_partition_keeps_correlated_rate_and_shape_inputs(monkeypatch):
    ttbar = "TTTo2L2Nu_centralUL16"
    remainder = "ttllJet_privateUL16"
    original_group_map = {"ttll": [ttbar, remainder], "Flips": []}
    hist_obj = _make_process_sparse_hist(
        {
            ttbar: {"nominal": 5.0, "renormUp": 6.0, "renormDown": 4.0,
                    "factUp": 7.0, "factDown": 3.0},
            remainder: {"nominal": 7.0, "renormUp": 9.0, "renormDown": 5.0,
                        "factUp": 10.0, "factDown": 4.0},
        }
    )
    monkeypatch.setattr(make_cr_and_sr_plots, "SR_GRP_MAP", original_group_map)
    rate_before = make_cr_and_sr_plots.get_rate_syst_arrs(
        hist_obj, original_group_map, group_type="SR"
    )
    shape_before, details_before = make_cr_and_sr_plots.get_shape_syst_arrs(
        hist_obj, group_type="SR", return_details=True
    )
    assert set(details_before["valid_bases"]) == {"renorm", "fact"}

    draw_view = make_cr_and_sr_plots._resolve_sr_signed_display_view(
        hist_obj.integrate("systematic", "nominal"),
        {"ttbar": [ttbar], "ttll": [remainder]},
    )
    assert draw_view is None  # Positive groups need no signed fallback.
    assert original_group_map == make_cr_and_sr_plots.SR_GRP_MAP
    rate_after = make_cr_and_sr_plots.get_rate_syst_arrs(
        hist_obj, original_group_map, group_type="SR"
    )
    shape_after, details_after = make_cr_and_sr_plots.get_shape_syst_arrs(
        hist_obj, group_type="SR", return_details=True
    )
    assert details_after == details_before
    for before, after in zip(rate_before + shape_before, rate_after + shape_after):
        np.testing.assert_allclose(before, after)


def test_sr_jes_selection_is_per_process_era_and_matches_card():
    source = {
        "ttH_central2022": {
            "nominal": 10.0,
            "JES_TotalUp": 100.0, "JES_TotalDown": 1.0,
            "JES_AbsoluteUp": 12.0, "JES_AbsoluteDown": 8.0,
        },
        "ttH_central2022EE": {
            "nominal": 20.0,
            "JES_TotalUp": 50.0, "JES_TotalDown": 2.0,
            "JES_FlavorQCDUp": 23.0, "JES_FlavorQCDDown": 17.0,
        },
        "ttH_central2023": {
            "nominal": 5.0,
            "JES_TotalUp": 9.0, "JES_TotalDown": 1.0,
        },
    }
    hist_obj = _make_process_sparse_hist(source)
    (low, high), details = make_cr_and_sr_plots.get_shape_syst_arrs(
        hist_obj, group_type="SR", return_details=True,
    )
    assert details["valid_bases"] == ("JES_Absolute", "JES_FlavorQCD")
    without_total = {
        process: {name: weight for name, weight in weights.items()
                  if not name.startswith("JES_Total")}
        for process, weights in source.items()
    }
    expected, _ = make_cr_and_sr_plots.get_shape_syst_arrs(
        _make_process_sparse_hist(without_total), group_type="SR", return_details=True,
    )
    for actual, reference in zip((low, high), expected):
        np.testing.assert_allclose(actual, reference)


def test_sr_only_total_does_not_create_card_jes_nuisance():
    hist_obj = _make_process_sparse_hist({
        "ttH_central2022": {
            "nominal": 10.0, "JES_TotalUp": 15.0, "JES_TotalDown": 5.0,
        },
    })
    result, details = make_cr_and_sr_plots.get_shape_syst_arrs(
        hist_obj, group_type="SR", return_details=True,
    )
    assert details["valid_bases"] == ()
    nominal_only = _make_process_sparse_hist({"ttH_central2022": {"nominal": 10.0}})
    expected = make_cr_and_sr_plots.get_shape_syst_arrs(nominal_only, group_type="SR")
    for actual, reference in zip(result, expected):
        np.testing.assert_allclose(actual, reference)


def test_sr_incomplete_component_pair_is_not_hidden_by_total():
    hist_obj = _make_process_sparse_hist({
        "ttH_central2022": {
            "nominal": 10.0, "JES_TotalUp": 15.0,
            "JES_AbsoluteUp": 12.0,
        },
    })
    with pytest.raises(ValueError, match="Incomplete SR variation pair.*JES_Absolute"):
        make_cr_and_sr_plots.get_shape_syst_arrs(hist_obj, group_type="SR")


def test_sr_nominal_like_tau_pair_is_kept_when_card_option_is_off():
    from analysis.topeft_run2.make_cards import build_arg_parser as card_parser

    assert card_parser().parse_args([]).suppress_nominal_like_tau_shapes is False
    source = "CMS_fake_t_DeepTau2018v2p5_VSmu_abseta0to0p4_2022"
    hist_obj = _make_process_sparse_hist({
        "ttH_central2022": {
            "nominal": 10.0,
            source + "Up": 10.00001,
            source + "Down": 9.99999,
        },
    })
    (_, high), details = make_cr_and_sr_plots.get_shape_syst_arrs(
        hist_obj, group_type="SR", return_details=True,
    )
    assert source in details["valid_bases"]
    assert high[0] > 0


def test_sr_non_njets_special_card_rates_fail_closed():
    hist_obj = _make_process_sparse_hist({
        "WZTo3LNu_central2022": {"nominal": 10.0},
    })
    with pytest.raises(ValueError, match="channel/jet-specific diboson_njets"):
        make_cr_and_sr_plots.get_rate_syst_arrs(
            hist_obj, {}, group_type="SR", variable="lj0pt",
        )


def test_sr_non_njets_rates_sum_signed_physical_channel_responses(monkeypatch):
    hist_obj = make_cr_and_sr_plots.SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="met"),
    )
    for channel, weight in (("3l_onZ_1b_2j", 5.0), ("3l_onZ_1b_3j", 7.0)):
        hist_obj.fill(
            process="WZTo3LNu_central2022", channel=channel,
            systematic="nominal", met=0.5, weight=weight,
        )
    seen = []

    def physical_rate(process, values, *, channel, variable):
        seen.append((process, channel, variable, values.copy()))
        if channel.endswith("_2j"):
            return {"diboson_njets": (values * -0.2, values * 0.2)}
        return {"diboson_njets": (values * (2 / 7), values * (-2 / 7))}

    monkeypatch.setattr(make_cr_and_sr_plots, "_sr_rate_responses", physical_rate)
    down_square, up_square = make_cr_and_sr_plots.get_rate_syst_arrs(
        hist_obj, {}, group_type="SR", variable="lj0pt",
        channel_bins=("3l_onZ_1b_2j", "3l_onZ_1b_3j"),
    )
    assert [row[1] for row in seen] == ["3l_onZ_1b_2j", "3l_onZ_1b_3j"]
    np.testing.assert_allclose([row[3][0] for row in seen], [5.0, 7.0])
    # The two physical contributions cancel only after signed, correlated sum.
    np.testing.assert_allclose(down_square, [1.0])
    np.testing.assert_allclose(up_square, [1.0])


def test_sr_non_njets_mixed_card_factors_ignore_display_partition(monkeypatch):
    from topeft.modules.plot_uncertainty_contract import rate_responses

    physical = (
        ("WZTo3LNu_central2022", "3l_onZ_1b_2j", 5.0, 0.86),
        ("ZZTo4L_central2022", "3l_onZ_1b_3j", 7.0, 1.43),
    )
    h_mc = make_cr_and_sr_plots.SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="lj0pt"),
    )
    for process, channel, value, high in physical:
        h_mc.fill(process=process, channel=channel, systematic="nominal",
                  lj0pt=0.5, weight=value)
        down, up = rate_responses(
            process, np.array([value]), channel=channel, variable="lj0pt",
        )["diboson_njets"]
        np.testing.assert_allclose(down, [value * ((2 - high) - 1)])
        np.testing.assert_allclose(up, [value * (high - 1)])

    def only_card_jet_rate(process, values, *, channel, variable):
        return {"diboson_njets": rate_responses(
            process, values, channel=channel, variable=variable,
        )["diboson_njets"]}

    monkeypatch.setattr(make_cr_and_sr_plots, "_sr_rate_responses", only_card_jet_rate)
    merged = {"one label": [item[0] for item in physical]}
    split = {"other label": [physical[1][0]], "first label": [physical[0][0]]}
    observed = []
    for group_map, selected_channels in (
        (merged, (physical[0][1], physical[1][1])),
        (split, (physical[1][1], physical[0][1])),
    ):
        observed.append(make_cr_and_sr_plots.get_rate_syst_arrs(
            h_mc, group_map, group_type="SR", variable="lj0pt",
            channel_bins=selected_channels,
        ))
    expected_signed = 5.0 * (0.86 - 1) + 7.0 * (1.43 - 1)
    assert expected_signed == pytest.approx(2.31)
    for down_square, up_square in observed:
        np.testing.assert_allclose(down_square, [expected_signed ** 2])
        np.testing.assert_allclose(up_square, [expected_signed ** 2])


def test_metadata_skipped_group_is_not_reintroduced_for_systematics(monkeypatch):
    hist_obj = _make_process_sparse_hist(
        {
            "ttbar_central2022": {
                "nominal": 20.0,
                "renormUp": 24.0,
                "renormDown": 16.0,
            },
            "WJetsToLNu_centralUL17": {
                "nominal": 100.0,
                "renormUp": 150.0,
                "renormDown": 50.0,
            },
        }
    )
    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "CR_GRP_MAP",
        {"Top": ["ttbar_central2022"]},
    )

    (shape_m, shape_p), details = make_cr_and_sr_plots.get_shape_syst_arrs(
        hist_obj,
        group_type="CR",
        return_details=True,
    )

    assert details["valid_bases"] == ("renorm",)
    assert details["skipped_failed"] == ()
    assert np.allclose(shape_p, np.array([4.0**2]))
    assert np.allclose(shape_m, np.array([4.0**2]))
