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
