import json
import re
from pathlib import Path

import pytest

from analysis.topeft_run2 import analysis_processor as ap


def test_run_cr_requests_ptz_for_the_3l_cr_and_forwards_hist_variables():
    run_cr_source = (
        Path(__file__).resolve().parents[1]
        / "analysis"
        / "topeft_run2"
        / "run_cr.sh"
    ).read_text()
    full_run_source = (
        Path(__file__).resolve().parents[1]
        / "analysis"
        / "topeft_run2"
        / "fullR3_run.sh"
    ).read_text()

    assert 'read -r -a vars <<< "${var_set}"' in run_cr_source
    assert 'local cats=("$@")' in run_cr_source
    assert '--hist-vars "${vars[@]}"' in run_cr_source
    assert '--category-groups "${cats[@]}"' in run_cr_source
    assert 'HIST_VARS+=("$1")' in full_run_source
    assert 'HIST_LIST_ARGS=(--hist-list "${HIST_VARS[@]}")' in full_run_source


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


@pytest.mark.parametrize("all_analysis", [False, True])
def test_ptz_wtau_gating_allows_only_tau_onz_and_dy_tautau_cr(all_analysis):
    processor = ap.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=[],
        tau_h_analysis=not all_analysis,
        all_analysis=all_analysis,
    )

    fill_channels = [
        "2lss_p_1tau_onZ",
        "2lss_m_1tau_onZ",
        "1l_dy_tautau_CR",
    ]
    skip_channels = [
        "2lss_p_1tau_offZ",
        "2lss_m_1tau_offZ",
        "1l_1tau_CR",
        "2los_1tau_Ftau",
        "2los_1tau_Ttau",
        "2los_1tau_0b",
        "2los_CRZ",
        "2lss_CRflip",
        "3l_1tau_1b",
    ]

    for lep_chan in fill_channels:
        assert (
            processor._should_skip_histogram_fill(
                dense_axis_name="ptz_wtau",
                ch_name=lep_chan,
                lep_chan=lep_chan,
            )
            is False
        )

    for lep_chan in skip_channels:
        assert (
            processor._should_skip_histogram_fill(
                dense_axis_name="ptz_wtau",
                ch_name=lep_chan,
                lep_chan=lep_chan,
            )
            is True
        )


def test_2lss_1tau_onz_channels_require_onz_tau_selection():
    channel_path = Path(__file__).resolve().parents[1] / "topeft" / "channels" / "ch_lst.json"
    channel_data = json.loads(channel_path.read_text())

    for block_name in ("TAU_CH_LST_SR", "ALL_CH_LST_SR"):
        channels = channel_data[block_name]["2lss_1tau"]["lep_chan_lst"]

        onz_channels = [channel for channel in channels if channel[0].endswith("_onZ")]
        offz_channels = [channel for channel in channels if channel[0].endswith("_offZ")]

        assert onz_channels
        assert offz_channels
        for channel in onz_channels:
            assert "onZ_tau" in channel
            assert "offZ_tau" not in channel
        for channel in offz_channels:
            assert "offZ_tau" in channel
            assert "onZ_tau" not in channel


@pytest.mark.parametrize("all_analysis", [False, True])
def test_plain_ptz_cr_policy_fills_zll_and_diagnostic_crs(all_analysis):
    processor = ap.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=[],
        tau_h_analysis=not all_analysis,
        all_analysis=all_analysis,
    )

    for lep_chan in (
        "2los_CRZ",
        "2lss_CRflip",
        "2los_1tau_Ftau",
        "2los_1tau_Ttau",
        "2los_1tau_0b",
        "3l_CR",
    ):
        assert (
            processor._should_skip_histogram_fill(
                dense_axis_name="ptz",
                ch_name=f"{lep_chan}_0j",
                lep_chan=lep_chan,
            )
            is False
        )


@pytest.mark.parametrize("lep_chan", [
    "2lss_CR",
    "2los_CRtt",
    "1l_1tau_CR",
    "1l_dy_tautau_CR",
])
def test_plain_ptz_cr_policy_skips_non_zll_crs(lep_chan):
    processor = ap.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=[],
        tau_h_analysis=True,
    )

    assert (
        processor._should_skip_histogram_fill(
            dense_axis_name="ptz",
            ch_name=f"{lep_chan}_0j",
            lep_chan=lep_chan,
        )
        is True
    )


def test_plain_ptz_sr_policy_preserves_existing_onz_and_offz_decisions():
    tau_processor = ap.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=[],
        tau_h_analysis=True,
    )
    all_processor = ap.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=[],
        all_analysis=True,
    )

    assert (
        tau_processor._should_skip_histogram_fill(
            dense_axis_name="ptz",
            ch_name="3l_onZ_1b_2j",
            lep_chan="3l_onZ_1b",
        )
        is False
    )
    assert (
        tau_processor._should_skip_histogram_fill(
            dense_axis_name="ptz",
            ch_name="2los_onZ_1tau_3j",
            lep_chan="2los_onZ_1tau",
        )
        is False
    )
    for lep_chan in ("2lss_m_1tau_onZ", "2lss_p_1tau_onZ"):
        assert (
            tau_processor._should_skip_histogram_fill(
                dense_axis_name="ptz",
                ch_name=f"{lep_chan}_3j",
                lep_chan=lep_chan,
            )
            is True
        )
        assert (
            tau_processor._should_skip_histogram_fill(
                dense_axis_name="ptz_wtau",
                ch_name=f"{lep_chan}_3j",
                lep_chan=lep_chan,
            )
            is False
        )
    assert (
        tau_processor._should_skip_histogram_fill(
            dense_axis_name="ptz",
            ch_name="3l_m_offZ_1b_2j",
            lep_chan="3l_m_offZ_1b",
        )
        is True
    )
    assert (
        all_processor._should_skip_histogram_fill(
            dense_axis_name="ptz",
            ch_name="3l_m_offZ_low_1b_2j",
            lep_chan="3l_m_offZ_low_1b",
        )
        is False
    )
    assert (
        all_processor._should_skip_histogram_fill(
            dense_axis_name="ptz",
            ch_name="3l_m_offZ_none_1b_2j",
            lep_chan="3l_m_offZ_none_1b",
        )
        is True
    )


def test_plain_ptz_zll_cr_channel_definitions_are_explicitly_onz():
    channel_path = Path(__file__).resolve().parents[1] / "topeft" / "channels" / "ch_lst.json"
    channel_data = json.loads(channel_path.read_text())

    for block_name in ("CH_LST_CR", "TAU_CH_LST_CR"):
        assert "2l_onZ_as" in channel_data[block_name]["2l_CRflip"]["lep_chan_lst"][0]
        assert "2l_onZ" in channel_data[block_name]["2los_CRZ"]["lep_chan_lst"][0]


def test_plain_ptz_diagnostic_2los_1tau_crs_are_not_category_onz():
    channel_path = Path(__file__).resolve().parents[1] / "topeft" / "channels" / "ch_lst.json"
    channel_data = json.loads(channel_path.read_text())
    channels = channel_data["TAU_CH_LST_CR"]["2los_1tau"]["lep_chan_lst"]
    zero_b_channels = channel_data["TAU_CH_LST_CR"]["2los_1tau_0b"]["lep_chan_lst"]

    assert {channel[0] for channel in channels} == {
        "2los_1tau_Ftau",
        "2los_1tau_Ttau",
    }
    assert "2los_1tau_0b" not in {channel[0] for channel in channels}
    assert {channel[0] for channel in zero_b_channels} == {"2los_1tau_0b"}
    for channel in [*channels, *zero_b_channels]:
        assert "2los" in channel
        assert "2l_onZ" not in channel


@pytest.mark.parametrize("lep_chan", [
    "2los_CRZ",
    "2lss_CRflip",
    "2los_1tau_Ftau",
    "2los_1tau_Ttau",
    "2los_1tau_0b",
])
def test_ptz_wtau_regression_still_skips_plain_zll_and_diagnostic_crs(lep_chan):
    processor = ap.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=[],
        tau_h_analysis=True,
    )

    assert (
        processor._should_skip_histogram_fill(
            dense_axis_name="ptz_wtau",
            ch_name=f"{lep_chan}_0j",
            lep_chan=lep_chan,
        )
        is True
    )
