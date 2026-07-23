import contextlib
from collections import defaultdict

import hist
import pytest

from analysis.topeft_run2 import make_cr_and_sr_plots
from topeft.modules.yield_tools import YieldTools


def _make_sparse_met_hist(entries, *, include_appl=False, include_data=True):
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    channel_axis = hist.axis.StrCategory([], name="channel", growth=True)
    systematic_axis = hist.axis.StrCategory([], name="systematic", growth=True)
    met_axis = hist.axis.Regular(1, 0.0, 1.0, name="met")

    axes = [process_axis, channel_axis]
    if include_appl:
        axes.append(hist.axis.StrCategory([], name="appl", growth=True))
    axes.extend([systematic_axis, met_axis])

    histogram = make_cr_and_sr_plots.tc_sparseHist.SparseHist(*axes)
    setattr(histogram, "_sumw2", defaultdict(lambda: None))

    for entry in entries:
        fill_kwargs = {
            "process": entry["process"],
            "channel": entry["channel"],
            "systematic": "nominal",
            "met": 0.5,
            "weight": entry["weight"],
        }
        if include_appl:
            fill_kwargs["appl"] = entry["appl"]
        histogram.fill(**fill_kwargs)
        if include_data and not str(entry["process"]).startswith("data"):
            data_kwargs = dict(fill_kwargs)
            data_kwargs["process"] = "data2022"
            data_kwargs["weight"] = entry.get("data_weight", 1.0)
            histogram.fill(**data_kwargs)

    return histogram


def _make_sparse_njets_hist(entries, *, include_data=True):
    process_axis = hist.axis.StrCategory([], name="process", growth=True)
    channel_axis = hist.axis.StrCategory([], name="channel", growth=True)
    systematic_axis = hist.axis.StrCategory([], name="systematic", growth=True)
    njets_axis = hist.axis.Regular(3, 0.0, 3.0, name="njets")

    histogram = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        process_axis, channel_axis, systematic_axis, njets_axis
    )
    setattr(histogram, "_sumw2", defaultdict(lambda: None))

    for entry in entries:
        fill_kwargs = {
            "process": entry["process"],
            "channel": entry["channel"],
            "systematic": "nominal",
            "njets": entry.get("njets", 1.5),
            "weight": entry["weight"],
        }
        histogram.fill(**fill_kwargs)
        if include_data and not str(entry["process"]).startswith("data"):
            data_kwargs = dict(fill_kwargs)
            data_kwargs["process"] = "data2022"
            data_kwargs["weight"] = entry.get("data_weight", 1.0)
            histogram.fill(**data_kwargs)

    return histogram


def test_get_appl_sr_bin_supports_1l_categories():
    assert YieldTools().get_appl_sr_bin("1l_1tau_CR") == "isSR_1l"


def test_integrate_category_logs_failures(caplog, monkeypatch):
    def _raise_for_test(*args, **kwargs):
        raise RuntimeError("forced integration failure")

    monkeypatch.setattr(make_cr_and_sr_plots.yt, "integrate_out_appl", _raise_for_test)

    with caplog.at_level("WARNING"):
        integrated = make_cr_and_sr_plots._integrate_category(
            object(),
            "mystery_CR",
            {"channel": ["mystery_CR_0j"]},
            region_name="CR",
            var_name="met",
            hist_label="mc histogram",
        )

    assert integrated is None
    assert "Failed to integrate mc histogram" in caplog.text
    assert "hist_cat=mystery_CR" in caplog.text
    assert "var_name=met" in caplog.text
    assert "forced integration failure" in caplog.text


def test_merged_mode_renders_1l_tau_cr_alias_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(make_cr_and_sr_plots, "tc_make_html", lambda *_args, **_kwargs: None)

    histogram = _make_sparse_met_hist(
        [
            {
                "process": "ttH_central2022",
                "channel": "1l_e_1tau_CR_0j",
                "appl": "isSR_1l",
                "weight": 3.0,
            },
            {
                "process": "ttH_central2022",
                "channel": "1l_m_1tau_CR_2j",
                "appl": "isSR_1l",
                "weight": 4.0,
            },
            {
                "process": "ttH_central2022",
                "channel": "1l_e_dy_tautau_CR_0j",
                "appl": "isSR_1l",
                "weight": 5.0,
            },
            {
                "process": "ttH_central2022",
                "channel": "1l_m_dy_tautau_CR_1j",
                "appl": "isSR_1l",
                "weight": 6.0,
            },
        ],
        include_appl=True,
    )

    make_cr_and_sr_plots.run_plots_for_region(
        "CR",
        {"met": histogram},
        years=["2022"],
        save_dir_path=str(tmp_path),
        channel_output="merged",
        skip_syst_errs=True,
        workers=1,
        verbose=False,
        unblind=False,
    )

    assert (tmp_path / "cr_1l_1tau_tt" / "1l_1tau_CR_met.png").exists()
    assert (tmp_path / "cr_dy_tautau" / "1l_dy_tautau_CR_met.png").exists()


def test_merged_njets_preserves_empty_low_bins_for_1l_tau_cr(tmp_path, monkeypatch):
    monkeypatch.setattr(make_cr_and_sr_plots, "tc_make_html", lambda *_args, **_kwargs: None)

    histogram = _make_sparse_met_hist(
        [
            {
                "process": "ttH_central2022",
                "channel": "1l_e_1tau_CR_0j",
                "weight": 0.0,
            },
            {
                "process": "ttH_central2022",
                "channel": "1l_e_1tau_CR_1j",
                "weight": 0.0,
            },
            {
                "process": "ttH_central2022",
                "channel": "1l_e_1tau_CR_2j",
                "weight": 4.0,
            },
            {
                "process": "ttH_central2022",
                "channel": "1l_e_dy_tautau_CR_0j",
                "weight": 5.0,
            },
            {
                "process": "ttH_central2022",
                "channel": "1l_e_dy_tautau_CR_1j",
                "weight": 6.0,
            },
            {
                "process": "ttH_central2022",
                "channel": "1l_e_dy_tautau_CR_2j",
                "weight": 7.0,
            },
        ],
        include_appl=False,
    )

    make_cr_and_sr_plots.run_plots_for_region(
        "CR",
        {"met": histogram},
        years=["2022"],
        save_dir_path=str(tmp_path),
        channel_output="merged-njets",
        skip_syst_errs=True,
        workers=1,
        verbose=False,
        unblind=False,
    )

    assert not (tmp_path / "cr_1l_1tau_tt_0j" / "1l_1tau_CR_0j_met.png").exists()
    assert not (tmp_path / "cr_1l_1tau_tt_1j" / "1l_1tau_CR_1j_met.png").exists()
    assert (tmp_path / "cr_1l_1tau_tt_2j" / "1l_1tau_CR_2j_met.png").exists()

    assert (tmp_path / "cr_dy_tautau_0j" / "1l_dy_tautau_CR_0j_met.png").exists()
    assert (tmp_path / "cr_dy_tautau_1j" / "1l_dy_tautau_CR_1j_met.png").exists()
    assert (tmp_path / "cr_dy_tautau_2j" / "1l_dy_tautau_CR_2j_met.png").exists()


@pytest.mark.parametrize(
    ("channel_output", "warning_ctx"),
    [
        ("merged", contextlib.nullcontext()),
        ("merged-njets", contextlib.nullcontext()),
        (
            "both",
            pytest.warns(
                RuntimeWarning, match="Skipping split channel output for CR"
            ),
        ),
        (
            "both-njets",
            pytest.warns(
                RuntimeWarning, match="Skipping split channel output for CR"
            ),
        ),
    ],
)
def test_aggregate_njets_modes_render_base_channel_outputs(
    tmp_path, monkeypatch, channel_output, warning_ctx
):
    monkeypatch.setattr(make_cr_and_sr_plots, "tc_make_html", lambda *_args, **_kwargs: None)

    histogram = _make_sparse_njets_hist(
        [
            {
                "process": "ttH_central2022",
                "channel": "1l_1tau_CR",
                "njets": 0.5,
                "weight": 3.0,
            },
            {
                "process": "ttH_central2022",
                "channel": "1l_dy_tautau_CR",
                "njets": 1.5,
                "weight": 4.0,
            },
        ]
    )

    with warning_ctx:
        make_cr_and_sr_plots.run_plots_for_region(
            "CR",
            {"njets": histogram},
            years=["2022"],
            save_dir_path=str(tmp_path),
            channel_output=channel_output,
            skip_syst_errs=True,
            workers=1,
            verbose=False,
            unblind=False,
        )

    assert (tmp_path / "cr_1l_1tau_tt" / "1l_1tau_CR_njets.png").exists()
    assert (tmp_path / "cr_dy_tautau" / "1l_dy_tautau_CR_njets.png").exists()
    assert not (tmp_path / "cr_1l_1tau_tt_0j").exists()
    assert not (tmp_path / "cr_dy_tautau_0j").exists()


def test_merged_njets_skips_truly_empty_base_njets_categories(tmp_path, monkeypatch):
    monkeypatch.setattr(make_cr_and_sr_plots, "tc_make_html", lambda *_args, **_kwargs: None)

    histogram = _make_sparse_njets_hist(
        [
            {
                "process": "ttH_central2022",
                "channel": "1l_1tau_CR",
                "njets": 0.5,
                "weight": 0.0,
                "data_weight": 0.0,
            },
            {
                "process": "ttH_central2022",
                "channel": "1l_dy_tautau_CR",
                "njets": 1.5,
                "weight": 5.0,
            },
        ]
    )

    make_cr_and_sr_plots.run_plots_for_region(
        "CR",
        {"njets": histogram},
        years=["2022"],
        save_dir_path=str(tmp_path),
        channel_output="merged-njets",
        skip_syst_errs=True,
        workers=1,
        verbose=False,
        unblind=False,
    )

    assert not (tmp_path / "cr_1l_1tau_tt" / "1l_1tau_CR_njets.png").exists()
    assert (tmp_path / "cr_dy_tautau" / "1l_dy_tautau_CR_njets.png").exists()
