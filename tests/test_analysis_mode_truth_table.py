import re

import pytest

from analysis.topeft_run2 import analysis_processor as ap


@pytest.mark.parametrize(
    ("offz", "tau", "fwd", "all_mode"),
    [
        (True, True, False, False),
        (True, False, True, False),
        (False, True, True, False),
        (True, False, False, True),
        (False, True, False, True),
        (False, False, True, True),
    ],
)
def test_mode_flags_reject_conflicting_combinations(offz, tau, fwd, all_mode):
    with pytest.raises(ValueError, match=re.escape(ap.ANALYSIS_MODE_EXCLUSIVE_ERROR)):
        ap.validate_analysis_mode_flags(offz, tau, fwd, all_mode)


@pytest.mark.parametrize(
    ("offz", "tau", "fwd", "all_mode", "expected_sr", "expected_cr", "enable_offz", "enable_tau", "enable_fwd"),
    [
        (False, False, False, False, "TOP22_006_CH_LST_SR", "CH_LST_CR", False, False, False),
        (True, False, False, False, "OFFZ_SPLIT_CH_LST_SR", "CH_LST_CR", True, False, False),
        (False, True, False, False, "TAU_CH_LST_SR", "TAU_CH_LST_CR", False, True, False),
        (False, False, True, False, "FWD_CH_LST_SR", "CH_LST_CR", False, False, True),
        (False, False, False, True, "ALL_CH_LST_SR", "TAU_CH_LST_CR", True, True, True),
    ],
)
def test_mode_truth_table_helpers(
    offz,
    tau,
    fwd,
    all_mode,
    expected_sr,
    expected_cr,
    enable_offz,
    enable_tau,
    enable_fwd,
):
    resolved_flags = ap.validate_analysis_mode_flags(offz, tau, fwd, all_mode)
    sr_name, cr_name = ap.resolve_category_dict_names(
        resolved_flags["offz_3l_split"],
        resolved_flags["tau_h_analysis"],
        resolved_flags["fwd_analysis"],
        resolved_flags["all_analysis"],
    )
    toggles = ap.derive_analysis_enable_toggles(
        resolved_flags["offz_3l_split"],
        resolved_flags["tau_h_analysis"],
        resolved_flags["fwd_analysis"],
        resolved_flags["all_analysis"],
    )

    assert sr_name == expected_sr
    assert cr_name == expected_cr
    assert toggles["enable_offz_blocks"] is enable_offz
    assert toggles["enable_tau_blocks"] is enable_tau
    assert toggles["enable_fwd_blocks"] is enable_fwd


def test_analysis_processor_safety_net_rejects_conflicting_modes():
    with pytest.raises(ValueError, match=re.escape(ap.ANALYSIS_MODE_EXCLUSIVE_ERROR)):
        ap.AnalysisProcessor(
            samples={},
            wc_names_lst=[],
            hist_lst=[],
            offZ_split=True,
            tau_h_analysis=True,
        )


def test_analysis_processor_all_mode_enables_all_blocks():
    processor = ap.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=[],
        all_analysis=True,
    )

    assert processor.sr_category_dict_name == "ALL_CH_LST_SR"
    assert processor.cr_category_dict_name == "TAU_CH_LST_CR"
    assert processor.enable_offz_blocks is True
    assert processor.enable_tau_blocks is True
    assert processor.enable_fwd_blocks is True


def test_all_mode_keeps_offz_split_ptz_histograms():
    processor = ap.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=[],
        all_analysis=True,
    )
    should_skip = processor._should_skip_histogram_fill(
        dense_axis_name="ptz",
        ch_name="3l_channel",
        lep_chan="3l_m_offZ_low_1b",
    )
    assert should_skip is False


def test_tau_mode_ptz_wtau_gating_is_strict():
    processor = ap.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=[],
        tau_h_analysis=True,
    )
    assert (
        processor._should_skip_histogram_fill(
            dense_axis_name="ptz_wtau",
            ch_name="2l_channel",
            lep_chan="2lss_p_1tau_onZ",
        )
        is False
    )
    assert (
        processor._should_skip_histogram_fill(
            dense_axis_name="ptz_wtau",
            ch_name="2l_channel",
            lep_chan="2lss_p_1tau_offZ",
        )
        is True
    )
