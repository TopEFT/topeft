from collections import OrderedDict
import concurrent.futures
from types import SimpleNamespace

import hist
import pytest

from analysis.topeft_run2 import make_cr_and_sr_plots


EXACT_OFFZ_LABELS = (
    "3l_p_offZ_low_1b_2j",
    "3l_p_offZ_high_1b_2j",
    "3l_p_offZ_none_1b_2j",
    "3l_p_offZ_low_2b_5j",
    "3l_p_offZ_high_2b_5j",
    "3l_p_offZ_none_2b_5j",
)


def _make_channel_hist(channels):
    histogram = hist.Hist(
        hist.axis.StrCategory(channels, name="channel"),
        hist.axis.Regular(1, 0.0, 1.0, name="observable"),
        storage=hist.storage.Double(),
    )
    for channel in channels:
        histogram.fill(channel=channel, observable=0.5, weight=1.0)
    return histogram


def _normalized_presets(raw_presets):
    normalized_input = OrderedDict()
    for preset_name, subgroups in raw_presets:
        normalized_input[preset_name] = OrderedDict(
            (
                subgroup_name,
                {"lep_chan_lst": [[base_name] for base_name in base_names]},
            )
            for subgroup_name, base_names in subgroups
        )
    return make_cr_and_sr_plots._normalize_producer_channel_presets(
        normalized_input
    )


def _resolve_sr(labels, *, presets=None, category_map=None, known_channels=None):
    return make_cr_and_sr_plots._resolve_producer_channel_fallback(
        region="SR",
        variable="ptz",
        primary_namespace="SR_CHAN_DICT",
        primary_known_channels=(
            make_cr_and_sr_plots.SR_KNOWN_CHANNELS
            if known_channels is None
            else known_channels
        ),
        active_yaml_category_map=(
            make_cr_and_sr_plots.SR_CHAN_DICT
            if category_map is None
            else category_map
        ),
        observed_channel_labels=labels,
        transformations=[],
        producer_presets=presets,
    )


def _minimal_sr_payload_context(histogram):
    return SimpleNamespace(
        name="SR",
        dict_of_hists={"ptz": histogram},
        skip_sparse_2d=False,
        preserve_njets_bins=False,
        channel_rules={"default": [], "variables": {}, "conditional": []},
        channel_map=make_cr_and_sr_plots.SR_CHAN_DICT,
        channel_base_to_alias=make_cr_and_sr_plots.SR_CHAN_ALIASES,
        channel_dict_name="SR_CHAN_DICT",
        channels_split_by_lepflav=False,
        channel_mode="per-channel",
        is_lepton_flavor_in_pkl=False,
        channel_output_mode="merged",
    )


def test_yaml_only_path_preserves_primary_map_and_needs_no_fallback():
    observed = ("2lss_p_4j",)
    original_map = OrderedDict(
        (name, list(labels))
        for name, labels in make_cr_and_sr_plots.SR_CHAN_DICT.items()
    )

    resolution = _resolve_sr(observed)

    assert resolution is None
    assert make_cr_and_sr_plots.SR_CHAN_DICT == original_map
    make_cr_and_sr_plots.validate_variable_channel_coverage(
        [_make_channel_hist(observed)],
        make_cr_and_sr_plots.SR_KNOWN_CHANNELS,
        [],
        region="SR",
        variable="ptz",
        region_dict_name="SR_CHAN_DICT",
    )


def test_exact_offz_fallback_selects_all_preset_and_survives_payload_filters():
    histogram = _make_channel_hist(EXACT_OFFZ_LABELS)
    region_ctx = _minimal_sr_payload_context(histogram)
    original_region_map = OrderedDict(
        (name, list(labels)) for name, labels in region_ctx.channel_map.items()
    )

    payload = make_cr_and_sr_plots._prepare_variable_payload(
        "ptz", region_ctx, metadata_only=True
    )
    resolution = payload["channel_fallback_resolution"]

    assert resolution["selected_preset"] == "ALL_CH_LST_SR"
    assert resolution["fallback_observed_labels"] == EXACT_OFFZ_LABELS
    assert resolution["augmented_categories"] == ("3l_p_offZ",)
    assert payload["channel_dict"]["3l_p_offZ"] == list(EXACT_OFFZ_LABELS)
    assert region_ctx.channel_map == original_region_map
    grouped_labels = {
        label
        for channel_labels in payload["channel_dict"].values()
        for label in channel_labels or ()
    }
    assert set(EXACT_OFFZ_LABELS).issubset(grouped_labels)

    score_by_name = {
        score["preset_name"]: score for score in resolution["candidate_scores"]
    }
    assert (
        score_by_name["ALL_CH_LST_SR"]["category_compatibility"]
        > score_by_name["OFFZ_SPLIT_CH_LST_SR"]["category_compatibility"]
    )

    validation_payload = {
        "hist_mc": histogram,
        "hist_data": None,
        "channel_transformations": [],
        "channel_fallback_resolution": resolution,
    }
    make_cr_and_sr_plots._ensure_variable_channel_coverage_validated(
        "ptz", region_ctx, validation_payload
    )
    assert validation_payload["_global_channel_coverage_validated"] is True


def test_fallback_assigns_both_offz_charges_to_distinct_subgroups():
    labels = (
        "3l_m_offZ_low_1b_2j",
        "3l_m_offZ_none_2b_5j",
        "3l_p_offZ_high_1b_2j",
        "3l_p_offZ_none_2b_5j",
    )

    resolution = _resolve_sr(labels, known_channels=())

    assert resolution["selected_preset"] == "ALL_CH_LST_SR"
    assert resolution["augmented_category_map"]["3l_m_offZ"] == list(labels[:2])
    assert resolution["augmented_category_map"]["3l_p_offZ"] == list(labels[2:])


def test_no_union_of_incoherent_presets_is_accepted():
    presets = _normalized_presets(
        [
            ("ONLY_A_CH_LST_SR", [("group_a", ("unknown_a",))]),
            ("ONLY_B_CH_LST_SR", [("group_b", ("unknown_b",))]),
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        _resolve_sr(
            ("unknown_a_2j", "unknown_b_3j"),
            presets=presets,
            category_map=OrderedDict(),
            known_channels=(),
        )

    message = str(exc_info.value)
    assert "No single coherent producer preset" in message
    assert "ONLY_A_CH_LST_SR" in message
    assert "ONLY_B_CH_LST_SR" in message
    assert "unknown_a_2j" in message
    assert "unknown_b_3j" in message


def test_typo_remains_fatal_with_checked_preset_diagnostics():
    typo = "3l_p_offZ_hihg_1b_2j"

    with pytest.raises(ValueError) as exc_info:
        _resolve_sr((typo,))

    message = str(exc_info.value)
    assert "region 'SR', variable 'ptz'" in message
    assert "primary_namespace=SR_CHAN_DICT" in message
    assert "compatible_presets_checked" in message
    assert "per_preset_coverage_summary" in message
    assert "unresolved_labels" in message
    assert typo in message


def test_region_compatible_presets_never_cross_sr_cr_boundary():
    presets = _normalized_presets(
        [
            ("BASIC_CH_LST_SR", [("sr", ("sr_base",))]),
            ("CH_LST_CR", [("cr", ("cr_base",))]),
            ("TAU_CH_LST_CR", [("tau_cr", ("tau_cr_base",))]),
            ("ANOTHER_CH_LST_SR", [("sr_two", ("sr_two_base",))]),
        ]
    )

    sr_names = make_cr_and_sr_plots._region_compatible_producer_preset_names(
        "SR", presets
    )
    cr_names = make_cr_and_sr_plots._region_compatible_producer_preset_names(
        "CR", presets
    )

    assert sr_names == ("BASIC_CH_LST_SR", "ANOTHER_CH_LST_SR")
    assert cr_names == ("CH_LST_CR", "TAU_CH_LST_CR")


def test_equal_candidate_scores_use_stable_preset_name_tie_break():
    presets = _normalized_presets(
        [
            ("B_CH_LST_SR", [("compatible_group", ("new_base",))]),
            ("A_CH_LST_SR", [("compatible_group", ("new_base",))]),
        ]
    )

    resolution = _resolve_sr(
        ("new_base_2j",),
        presets=presets,
        category_map=OrderedDict([("compatible_group_existing", [])]),
        known_channels=(),
    )

    assert resolution["selected_preset"] == "A_CH_LST_SR"
    scores = resolution["candidate_scores"]
    assert scores[0]["category_compatibility"] == scores[1]["category_compatibility"]
    assert scores[0]["observed_base_coverage"] == scores[1]["observed_base_coverage"]
    assert scores[0]["unrelated_producer_bases"] == scores[1]["unrelated_producer_bases"]


def test_selection_prefers_coverage_of_all_observed_bases_after_compatibility():
    presets = _normalized_presets(
        [
            (
                "UNKNOWN_ONLY_CH_LST_SR",
                [("compatible_group", ("new_base",))],
            ),
            (
                "ALL_OBSERVED_CH_LST_SR",
                [("compatible_group", ("new_base", "known_base"))],
            ),
        ]
    )

    resolution = _resolve_sr(
        ("new_base_2j", "known_base_3j"),
        presets=presets,
        category_map=OrderedDict([("compatible_group_existing", [])]),
        known_channels=("known_base_3j",),
    )

    assert resolution["selected_preset"] == "ALL_OBSERVED_CH_LST_SR"


def test_selection_prefers_fewer_unrelated_bases_after_equal_coverage():
    presets = _normalized_presets(
        [
            (
                "WIDE_CH_LST_SR",
                [("compatible_group", ("new_base", "unused_a", "unused_b"))],
            ),
            (
                "NARROW_CH_LST_SR",
                [("compatible_group", ("new_base", "unused_a"))],
            ),
        ]
    )

    resolution = _resolve_sr(
        ("new_base_2j",),
        presets=presets,
        category_map=OrderedDict([("compatible_group_existing", [])]),
        known_channels=(),
    )

    assert resolution["selected_preset"] == "NARROW_CH_LST_SR"


def test_cached_loader_returns_immutable_source_ordered_presets():
    make_cr_and_sr_plots._load_producer_channel_presets.cache_clear()

    first = make_cr_and_sr_plots._load_producer_channel_presets()
    second = make_cr_and_sr_plots._load_producer_channel_presets()

    assert first is second
    assert isinstance(first, tuple)
    assert all(isinstance(preset, tuple) for preset in first)
    assert first[0][0] == "TOP22_006_CH_LST_SR"


def test_parent_aggregates_failures_before_executor_construction(monkeypatch):
    region_ctx = SimpleNamespace(
        name="SR",
        dict_of_hists={"bad_a": object(), "bad_b": object()},
        apply_category_skips=False,
        skip_variables=set(),
        unblind_default=False,
    )
    executor_calls = []

    def _fail_prepare(var_name, *_args, **_kwargs):
        raise ValueError(
            "primary_namespace=SR_CHAN_DICT; unknown_labels=['{}_typo']; "
            "compatible_presets_checked=['ALL_CH_LST_SR']; "
            "per_preset_coverage_summary=none; unresolved_labels=['{}_typo']".format(
                var_name, var_name
            )
        )

    def _executor_must_not_run(*_args, **_kwargs):
        executor_calls.append(True)
        raise AssertionError("executor constructed before parent validation")

    monkeypatch.setattr(
        make_cr_and_sr_plots, "_prepare_variable_payload", _fail_prepare
    )
    monkeypatch.setattr(
        concurrent.futures, "ProcessPoolExecutor", _executor_must_not_run
    )

    with pytest.raises(ValueError) as exc_info:
        make_cr_and_sr_plots.produce_region_plots(
            region_ctx,
            None,
            None,
            "none",
            False,
            False,
            workers=2,
        )

    message = str(exc_info.value)
    assert "Parent channel-schema validation failed for region 'SR' before worker creation" in message
    assert "variable 'bad_a'" in message
    assert "variable 'bad_b'" in message
    assert "primary_namespace=SR_CHAN_DICT" in message
    assert "compatible_presets_checked" in message
    assert "unresolved_labels" in message
    assert executor_calls == []


def test_parent_validates_all_payloads_before_worker_dispatch(monkeypatch):
    labels = list(EXACT_OFFZ_LABELS)
    region_ctx = SimpleNamespace(
        name="SR",
        dict_of_hists={"ptz": object(), "lt": object()},
        apply_category_skips=False,
        skip_variables=set(),
        unblind_default=False,
    )
    events = []
    prepared_payloads = {}

    def _prepare(var_name, *_args, **_kwargs):
        payload = {
            "channel_dict": OrderedDict([("3l_p_offZ", list(labels))]),
            "channel_transformations": [],
            "is_sparse2d": False,
            "channel_display_labels": {"3l_p_offZ": "3l_p_offZ"},
            "available_channels": tuple(labels),
            "hist_mc": object(),
            "hist_data": None,
            "hist_mc_sumw2_orig": None,
            "channel_fallback_resolution": {
                "primary_namespace": "SR_CHAN_DICT",
                "selected_preset": "ALL_CH_LST_SR",
                "checked_presets": ("ALL_CH_LST_SR",),
                "fallback_observed_labels": tuple(labels),
                "augmented_categories": ("3l_p_offZ",),
                "unresolved_labels": (),
            },
        }
        prepared_payloads[var_name] = payload
        return payload

    def _validate(var_name, _region_ctx, payload):
        events.append(("validate", var_name))
        payload["_global_channel_coverage_validated"] = True

    class _CompletedFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class _RecordingExecutor:
        def __init__(self, *_args, **_kwargs):
            events.append(("executor", None))
            shared = make_cr_and_sr_plots._SHARED_VARIABLE_PAYLOADS
            assert shared is not None
            assert set(shared) == {"ptz", "lt"}
            assert all(
                payload["channel_dict"]["3l_p_offZ"] == labels
                for payload in shared.values()
            )
            assert all(
                payload["_global_channel_coverage_validated"]
                for payload in shared.values()
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, _callable, task_id, _payload):
            return _CompletedFuture((task_id, 0, 0, set(), []))

    monkeypatch.setattr(make_cr_and_sr_plots, "_prepare_variable_payload", _prepare)
    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "_ensure_variable_channel_coverage_validated",
        _validate,
    )
    monkeypatch.setattr(
        make_cr_and_sr_plots,
        "_emit_channel_fallback_diagnostic",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        make_cr_and_sr_plots.multiprocessing,
        "get_start_method",
        lambda **_kwargs: "fork",
    )
    monkeypatch.setattr(
        concurrent.futures, "ProcessPoolExecutor", _RecordingExecutor
    )
    monkeypatch.setattr(concurrent.futures, "as_completed", lambda futures: futures)

    result = make_cr_and_sr_plots.produce_region_plots(
        region_ctx,
        None,
        None,
        "none",
        False,
        False,
        workers=2,
    )

    assert result == []
    assert events[:2] == [("validate", "ptz"), ("validate", "lt")]
    assert events[2] == ("executor", None)
    assert all(
        payload["channel_dict"]["3l_p_offZ"] == labels
        for payload in prepared_payloads.values()
    )
