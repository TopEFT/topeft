from types import SimpleNamespace
import hist
import pytest

from analysis.topeft_run2 import make_cr_and_sr_plots


def _make_channel_hist(channels):
    histogram = hist.Hist(
        hist.axis.StrCategory(channels, name="channel"),
        hist.axis.Regular(1, 0.0, 1.0, name="observable"),
        storage=hist.storage.Double(),
    )
    for channel in channels:
        histogram.fill(channel=channel, observable=0.5, weight=1.0)
    return histogram


def _region_ctx(
    region,
    *,
    preserve_njets_bins=False,
    channel_output_mode="merged",
):
    region_upper = str(region).upper()
    if region_upper == "CR":
        return SimpleNamespace(
            name="CR",
            channel_map=make_cr_and_sr_plots.CR_CHAN_DICT,
            channel_base_to_alias=make_cr_and_sr_plots.CR_CHAN_ALIASES,
            channel_dict_name="CR_CHAN_DICT",
            preserve_njets_bins=preserve_njets_bins,
            channel_output_mode=channel_output_mode,
            is_lepton_flavor_in_pkl=True,
        )
    return SimpleNamespace(
        name="SR",
        channel_map=make_cr_and_sr_plots.SR_CHAN_DICT,
        channel_base_to_alias=make_cr_and_sr_plots.SR_CHAN_ALIASES,
        channel_dict_name="SR_CHAN_DICT",
        preserve_njets_bins=preserve_njets_bins,
        channel_output_mode=channel_output_mode,
        is_lepton_flavor_in_pkl=False,
    )


def test_global_channel_coverage_reports_variable_level_mismatch():
    histo = _make_channel_hist(["2lss_p_4j", "3l_p_offZ_1b_2j"])

    with pytest.raises(ValueError) as exc_info:
        make_cr_and_sr_plots.validate_variable_channel_coverage(
            [histo],
            {"2lss_p_4j"},
            [],
            region="SR",
            variable="lj0pt",
            region_dict_name="SR_CHAN_DICT",
        )

    msg = str(exc_info.value)
    assert "Global channel coverage mismatch" in msg
    assert "variable 'lj0pt'" in msg
    assert "3l_p_offZ_1b_2j" in msg
    assert "subgroup" not in msg.lower()


def test_subgroup_validation_ignores_unrelated_axis_channels_when_scoped():
    histo = _make_channel_hist(["2lss_4t_p_5j", "3l_p_offZ_1b_2j"])

    make_cr_and_sr_plots.validate_variable_channel_coverage(
        [histo],
        {"2lss_4t_p_5j", "3l_p_offZ_1b_2j"},
        [],
        region="SR",
        variable="lj0pt",
        region_dict_name="SR_CHAN_DICT",
    )

    # Subgroup validation receives subgroup-local channels only.
    make_cr_and_sr_plots.validate_channel_group(
        [histo],
        ["2lss_4t_p_5j"],
        [],
        region="SR",
        subgroup="2lss_4t_p_5j",
        variable="lj0pt",
        available_channels=["2lss_4t_p_5j"],
    )


def test_subgroup_validation_message_lists_only_subgroup_local_channels():
    histo = _make_channel_hist(["2lss_4t_p_5j", "3l_p_offZ_1b_2j"])

    with pytest.raises(ValueError) as exc_info:
        make_cr_and_sr_plots.validate_channel_group(
            [histo],
            ["2lss_4t_p_5j"],
            [],
            region="SR",
            subgroup="2lss_4t_p_5j",
            variable="lj0pt",
            available_channels=["2lss_4t_p_5j_alias"],
        )

    msg = str(exc_info.value)
    assert "Subgroup '2lss_4t_p_5j'" in msg
    assert "2lss_4t_p_5j_alias" in msg
    assert "3l_p_offZ_1b_2j" not in msg


def test_subgroup_validation_requires_explicit_available_channels():
    histo = _make_channel_hist(["2lss_4t_p_5j"])

    with pytest.raises(TypeError):
        make_cr_and_sr_plots.validate_channel_group(
            [histo],
            ["2lss_4t_p_5j"],
            [],
            region="SR",
            subgroup="2lss_4t_p_5j",
            variable="lj0pt",
        )


def test_global_validation_allows_sr_aggregated_channels_for_njets():
    histo = _make_channel_hist(["3l_p_offZ_1b", "3l_p_offZ_2b"])
    region_ctx = _region_ctx("SR")
    variable_payload = {
        "hist_mc": histo,
        "hist_data": None,
        "channel_transformations": ["njets"],
    }

    make_cr_and_sr_plots._ensure_variable_channel_coverage_validated(
        "njets", region_ctx, variable_payload
    )


def test_global_validation_remains_strict_for_non_njets_variables():
    histo = _make_channel_hist(["definitely_missing_channel_1j"])
    region_ctx = _region_ctx("SR")
    variable_payload = {
        "hist_mc": histo,
        "hist_data": None,
        "channel_transformations": [],
    }

    with pytest.raises(ValueError) as exc_info:
        make_cr_and_sr_plots._ensure_variable_channel_coverage_validated(
            "lj0pt", region_ctx, variable_payload
        )

    msg = str(exc_info.value)
    assert "variable 'lj0pt'" in msg
    assert "definitely_missing_channel_1j" in msg


def test_global_validation_rejects_unknown_sr_njets_base_without_widening():
    histo = _make_channel_hist(["unknown_sr_base_channel"])
    region_ctx = _region_ctx("SR")
    variable_payload = {
        "hist_mc": histo,
        "hist_data": None,
        "channel_transformations": ["njets"],
    }

    with pytest.raises(ValueError) as exc_info:
        make_cr_and_sr_plots._ensure_variable_channel_coverage_validated(
            "njets", region_ctx, variable_payload
        )

    msg = str(exc_info.value)
    assert "variable 'njets'" in msg
    assert "unknown_sr_base_channel" in msg


def test_global_validation_rejects_unknown_cr_transformed_channels():
    histo = _make_channel_hist(["unknown_cr_channel"])
    region_ctx = _region_ctx("CR")
    variable_payload = {
        "hist_mc": histo,
        "hist_data": None,
        "channel_transformations": ["lepflav", "njets"],
    }

    with pytest.raises(ValueError) as exc_info:
        make_cr_and_sr_plots._ensure_variable_channel_coverage_validated(
            "njets", region_ctx, variable_payload
        )

    msg = str(exc_info.value)
    assert "variable 'njets'" in msg
    assert "unknown_cr_channel" in msg


def test_global_validation_accepts_cr_merged_njets_base_channels():
    histo = _make_channel_hist(
        [
            "1l_1tau_CR",
            "1l_dy_tautau_CR",
            "2los_1tau_Ftau",
            "2los_1tau_Ttau",
            "2los_CRZ",
            "2los_CRtt",
            "2lss_CR",
            "2lss_CRflip",
            "3l_CR",
        ]
    )
    region_ctx = _region_ctx(
        "CR",
        preserve_njets_bins=True,
        channel_output_mode="merged-njets",
    )
    variable_payload = {
        "hist_mc": histo,
        "hist_data": None,
        "channel_transformations": ["njets", "lepflav"],
    }

    make_cr_and_sr_plots._ensure_variable_channel_coverage_validated(
        "njets", region_ctx, variable_payload
    )


def test_channel_namespace_accepts_legacy_and_object_entries():
    namespace = make_cr_and_sr_plots._build_channel_namespace(
        {
            "cat_legacy": ["cat_legacy_2j"],
            "cat_object": {"leaves": ["cat_object_2j"], "alias": "shared_alias"},
        },
        region_label="TEST_CHAN_DICT",
    )

    assert namespace["base_to_leaves"]["cat_legacy"] == ["cat_legacy_2j"]
    assert namespace["base_to_leaves"]["cat_object"] == ["cat_object_2j"]
    assert namespace["base_to_alias"]["cat_legacy"] is None
    assert namespace["base_to_alias"]["cat_object"] == "shared_alias"


def test_channel_namespace_rejects_leaf_overlap():
    with pytest.raises(ValueError, match="leaf overlap"):
        make_cr_and_sr_plots._build_channel_namespace(
            {
                "cat_a": ["shared_2j"],
                "cat_b": ["shared_2j"],
            },
            region_label="TEST_CHAN_DICT",
        )


def test_channel_namespace_rejects_subset_leaf_overlap():
    with pytest.raises(ValueError, match="leaf overlap"):
        make_cr_and_sr_plots._build_channel_namespace(
            {
                "cat_a": ["shared_2j", "unique_3j"],
                "cat_b": ["shared_2j"],
            },
            region_label="TEST_CHAN_DICT",
        )


def test_channel_namespace_rejects_alias_base_collision():
    with pytest.raises(ValueError, match="Alias/base collision"):
        make_cr_and_sr_plots._build_channel_namespace(
            {
                "cat_a": {"leaves": ["cat_a_2j"], "alias": "cat_b"},
                "cat_b": ["cat_b_2j"],
            },
            region_label="TEST_CHAN_DICT",
        )


def test_channel_namespace_allows_shared_alias():
    namespace = make_cr_and_sr_plots._build_channel_namespace(
        {
            "cat_a": {"leaves": ["cat_a_2j"], "alias": "merged"},
            "cat_b": {"leaves": ["cat_b_2j"], "alias": "merged"},
        },
        region_label="TEST_CHAN_DICT",
    )

    assert namespace["alias_to_bases"]["merged"] == ("cat_a", "cat_b")
    assert namespace["output_name_by_base"]["cat_a"] == "merged"
    assert namespace["output_name_by_base"]["cat_b"] == "merged"


def test_parse_lepflav_token_requires_second_token_when_region_uses_lepflav_in_pkl():
    token = make_cr_and_sr_plots._parse_lepflav_token_for_region(
        "2los_em_CRtt_2j",
        region_name="CR",
        is_lepton_flavor_in_pkl=True,
    )
    assert token == "em"


def test_parse_lepflav_token_rejects_missing_or_invalid_token_when_required():
    with pytest.raises(ValueError, match="REGION_CHANNEL_CONFIG.CR.is_lepton_flavor_in_pkl=true"):
        make_cr_and_sr_plots._parse_lepflav_token_for_region(
            "2lss_CR_1j",
            region_name="CR",
            is_lepton_flavor_in_pkl=True,
        )


def test_parse_lepflav_token_is_disabled_for_regions_without_lepflav_in_pkl():
    token = make_cr_and_sr_plots._parse_lepflav_token_for_region(
        "2lss_p_4j",
        region_name="SR",
        is_lepton_flavor_in_pkl=False,
    )
    assert token is None


def test_output_folder_njets_suffix_uses_real_bin_without_literal_marker():
    region_ctx = SimpleNamespace(
        is_lepton_flavor_in_pkl=True,
        channel_output_names={"2los_CRZ": "cr_2los_Z"},
        channel_output_mode="merged-njets",
    )

    merged_label = make_cr_and_sr_plots._resolve_output_category_name(
        region_ctx, "2los_CRZ_0j"
    )
    split_label = make_cr_and_sr_plots._resolve_output_category_name(
        region_ctx, "2los_CRZ_ee_0j"
    )

    assert merged_label == "cr_2los_Z_0j"
    assert split_label == "cr_2los_Z_ee_0j"
    assert "_Nj" not in merged_label
    assert "_Nj" not in split_label
