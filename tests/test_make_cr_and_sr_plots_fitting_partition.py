from collections import OrderedDict, defaultdict

import hist
import pytest

from analysis.topeft_run2 import make_cr_and_sr_plots


FORWARD_M_GROUP = OrderedDict(
    [
        (
            "3l_m_offZ_2b_fwd",
            [
                "3l_m_offZ_2b_fwd_1j",
                "3l_m_offZ_2b_fwd_2j",
                "3l_m_offZ_2b_fwd_3j",
                "3l_m_offZ_2b_fwd_4j",
            ],
        )
    ]
)


def _make_lt_histogram(channels):
    histogram = make_cr_and_sr_plots.tc_sparseHist.SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(12, 0.0, 600.0, name="lt"),
    )
    setattr(histogram, "_sumw2", defaultdict(lambda: None))
    for channel in channels:
        histogram.fill(
            process="ttH_central2022",
            channel=channel,
            systematic="nominal",
            lt=25.0,
            weight=1.0,
        )
        histogram.fill(
            process="data2022",
            channel=channel,
            systematic="nominal",
            lt=25.0,
            weight=1.0,
        )
    return histogram


def _build_context(*, binning_mode, channel_output_mode, preserve_njets_bins):
    channels = FORWARD_M_GROUP["3l_m_offZ_2b_fwd"]
    histogram = _make_lt_histogram(channels)
    context = make_cr_and_sr_plots.build_region_context(
        "SR",
        {"lt": histogram},
        years=["2022"],
        unblind=False,
        channel_mode_override="aggregate",
        preserve_njets_bins=preserve_njets_bins,
        channel_output_mode=channel_output_mode,
        binning_mode=binning_mode,
    )
    return context


def test_forward_3l_partition_uses_exact_edges_and_maximal_classes():
    partitioned, semantic_labels = (
        make_cr_and_sr_plots._partition_fitting_compatible_channel_groups(
            FORWARD_M_GROUP,
            "lt",
        )
    )

    assert partitioned == OrderedDict(
        [
            (
                "3l_m_offZ_2b_fwd__fitting_1j_members",
                ["3l_m_offZ_2b_fwd_1j"],
            ),
            (
                "3l_m_offZ_2b_fwd__fitting_2j_3j_4j_members",
                [
                    "3l_m_offZ_2b_fwd_2j",
                    "3l_m_offZ_2b_fwd_3j",
                    "3l_m_offZ_2b_fwd_4j",
                ],
            ),
        ]
    )
    assert semantic_labels == {
        "3l_m_offZ_2b_fwd__fitting_1j_members": "3l_m_offZ_2b_fwd",
        "3l_m_offZ_2b_fwd__fitting_2j_3j_4j_members": "3l_m_offZ_2b_fwd",
    }
    assert tuple(
        make_cr_and_sr_plots.resolve_axis_edges(
            "lt", mode="fitting", channel="3l_m_offZ_2b_fwd_1j"
        )
    ) == (0.0, 150.0, 250.0, 500.0)
    assert tuple(
        make_cr_and_sr_plots.resolve_axis_edges(
            "lt", mode="fitting", channel="3l_m_offZ_2b_fwd_2j"
        )
    ) == (0.0, 250.0, 400.0, 500.0)


def test_partition_never_crosses_preexisting_semantic_groups():
    groups = OrderedDict(
        [
            *FORWARD_M_GROUP.items(),
            (
                "3l_p_offZ_2b_fwd",
                [
                    "3l_p_offZ_2b_fwd_1j",
                    "3l_p_offZ_2b_fwd_2j",
                    "3l_p_offZ_2b_fwd_3j",
                    "3l_p_offZ_2b_fwd_4j",
                ],
            ),
        ]
    )

    partitioned, semantic_labels = (
        make_cr_and_sr_plots._partition_fitting_compatible_channel_groups(
            groups,
            "lt",
        )
    )

    assert len(partitioned) == 4
    for output_label, members in partitioned.items():
        semantic_group = semantic_labels[output_label]
        assert all(member.startswith(f"{semantic_group}_") for member in members)


def test_identical_axes_do_not_cross_any_semantic_boundary(monkeypatch):
    groups = OrderedDict(
        [
            ("3l_m_offZ_low_1b", ["3l_m_offZ_low_1b_2j"]),
            ("3l_p_offZ_low_1b", ["3l_p_offZ_low_1b_2j"]),
            ("3l_m_offZ_high_1b", ["3l_m_offZ_high_1b_2j"]),
            ("3l_m_offZ_none_1b", ["3l_m_offZ_none_1b_2j"]),
            ("3l_m_offZ_none_2b", ["3l_m_offZ_none_2b_2j"]),
            ("3l_onZ_1b", ["3l_onZ_1b_2j"]),
            ("3l_1tau_1b", ["3l_1tau_1b_2j"]),
            ("3l_m_offZ_1b_fwd", ["3l_m_offZ_1b_fwd_2j"]),
        ]
    )
    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "resolve_axis_edges",
        lambda *args, **kwargs: [0.0, 1.0, 2.0],
    )

    partitioned, semantic_labels = (
        make_cr_and_sr_plots._partition_fitting_compatible_channel_groups(
            groups,
            "lt",
        )
    )

    assert partitioned == groups
    assert semantic_labels == {group: group for group in groups}


def test_single_axis_group_preserves_existing_identity():
    partitioned, semantic_labels = (
        make_cr_and_sr_plots._partition_fitting_compatible_channel_groups(
            FORWARD_M_GROUP,
            "ptz",
        )
    )

    assert partitioned == FORWARD_M_GROUP
    assert semantic_labels == {"3l_m_offZ_2b_fwd": "3l_m_offZ_2b_fwd"}


def test_prepare_payload_partitions_only_merged_fitting_mode():
    fitting_context = _build_context(
        binning_mode="fitting",
        channel_output_mode="merged",
        preserve_njets_bins=False,
    )
    processing_context = _build_context(
        binning_mode="processing",
        channel_output_mode="merged",
        preserve_njets_bins=False,
    )
    merged_njets_context = _build_context(
        binning_mode="fitting",
        channel_output_mode="merged-njets",
        preserve_njets_bins=True,
    )

    fitting_payload = make_cr_and_sr_plots._prepare_variable_payload(
        "lt", fitting_context, metadata_only=True
    )
    processing_payload = make_cr_and_sr_plots._prepare_variable_payload(
        "lt", processing_context, metadata_only=True
    )
    merged_njets_payload = make_cr_and_sr_plots._prepare_variable_payload(
        "lt", merged_njets_context, metadata_only=True
    )

    fitting_keys = set(fitting_payload["channel_dict"])
    assert "3l_m_offZ_2b_fwd__fitting_1j_members" in fitting_keys
    assert "3l_m_offZ_2b_fwd__fitting_2j_3j_4j_members" in fitting_keys
    assert processing_payload["channel_dict"]["3l_m_offZ_2b_fwd"] == [
        "3l_m_offZ_2b_fwd_1j",
        "3l_m_offZ_2b_fwd_2j",
        "3l_m_offZ_2b_fwd_3j",
        "3l_m_offZ_2b_fwd_4j",
    ]
    for channel in FORWARD_M_GROUP["3l_m_offZ_2b_fwd"]:
        assert merged_njets_payload["channel_dict"][channel] == [channel]
    assert not any(
        "__fitting_" in key for key in merged_njets_payload["channel_dict"]
    )


def test_unresolved_fitting_axis_fails_closed_with_member_diagnostic():
    with pytest.raises(
        ValueError,
        match=(
            "Unable to resolve fitting axis for variable 'not_a_variable' member "
            "'3l_m_offZ_2b_fwd_1j' inside presentation group "
            "'3l_m_offZ_2b_fwd'"
        ),
    ):
        make_cr_and_sr_plots._partition_fitting_compatible_channel_groups(
            FORWARD_M_GROUP,
            "not_a_variable",
        )
