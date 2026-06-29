import hist
import numpy as np
import pytest

from analysis.topeft_run2 import make_cr_and_sr_plots


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

    skipped = {
        entry["base"]: tuple(entry.get("missing", ()))
        for entry in details["skipped_orphans"]
    }
    assert "Foo" in skipped
    assert skipped["Foo"] == ("FooDown",)


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
