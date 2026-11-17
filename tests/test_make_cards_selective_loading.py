"""Small artifact fixtures for the card consumer's selective PKL path."""

import json

import numpy as np
import pytest

from analysis.topeft_run2 import make_cards
from topcoffea.modules.histEFT import HistEFT
from tests.test_split_nominal_loader import (
    _eft, _policy_for_families, _scalar, _write_versioned,
)
from topeft.modules import datacard_tools, nominal_schema
from topeft.modules.histogram_artifact import metadata_sidecar_path
from topeft.modules.missing_parton_contract import (
    infer_legacy_missing_parton_layout,
    load_registry_payload_layout,
    resolve_missing_parton_response_index,
    validate_legacy_missing_parton_payload,
)
from topeft.modules.paths import topeft_path


FAMILIES = ("njets", "ptz_wtau")
TARGET = "3l_onZ_2b_4j"
OTHER = "3l_onZ_1b_2j"


@pytest.mark.parametrize("era,year", [("run2", "UL18"), ("run3", "2022")])
def test_legacy_missing_parton_loads_for_current_sr_registry(era, year):
    payload_path = topeft_path(f"data/missing_parton/missing_parton_{era}.root")
    layout = infer_legacy_missing_parton_layout(payload_path)
    assert layout.registry == "ALL_CH_LST_SR"
    assert layout.public_lengths["2los_onZ_1tau"] == 6
    values = validate_legacy_missing_parton_payload(
        payload_path, sr_registry=layout.registry,
    )
    assert len(values["2los_onZ_1tau"]) == 4
    assert resolve_missing_parton_response_index("2los_onZ_1tau_3j", 3) == 3
    assert resolve_missing_parton_response_index("2los_onZ_1tau_5j", 5) == 3
    assert resolve_missing_parton_response_index("3l_onZ_2b_5j", 5) == 5

    card = datacard_tools.DatacardMaker.__new__(datacard_tools.DatacardMaker)
    card.do_nuisance = True
    card.year_lst = [year]
    rate_systs = card.load_systematics(
        f"params/rate_systs_{era}.json", payload_path,
    )
    nuisance_name = card.missing_parton_nuisance_name_for_years(
        card.year_lst, payload_path=payload_path,
    )
    nuisance = rate_systs[nuisance_name]
    for process in ("tllq", "tHq"):
        np.testing.assert_allclose(
            nuisance.get_process(process)["2los_onZ_1tau"][3],
            1 + values["2los_onZ_1tau"][3],
        )
        assert len(nuisance.get_process(process)["2los_onZ_1tau"]) == 4
    assert nuisance.get_process("ttll") == "-"


@pytest.mark.parametrize("change", ["alias_length", "other_length", "tree_order", "tree_identity"])
def test_legacy_missing_parton_rejects_unrelated_layout_mismatch(tmp_path, change):
    import uproot

    layout = load_registry_payload_layout("ALL_CH_LST_SR")
    categories = list(layout.ordered_base_categories)
    if change == "tree_order":
        categories.reverse()
    path = tmp_path / "invalid_missing_parton.root"
    with uproot.recreate(path) as output:
        for category in categories:
            name = "unrelated_category" if change == "tree_identity" and category == categories[0] else category
            length = layout.public_lengths[category]
            if category == "2los_onZ_1tau":
                length = 5 if change == "alias_length" else 4
            elif change == "other_length" and category == "3l_onZ_2b":
                length += 1
            tree = output.mktree(name, {"tllq": "float64"})
            tree.extend({"tllq": np.zeros(length)})
    with pytest.raises(ValueError, match="No maintained missing-parton payload layout matches"):
        infer_legacy_missing_parton_layout(path)


def test_current_six_entry_missing_parton_layout_remains_valid(tmp_path):
    import uproot

    layout = load_registry_payload_layout("ALL_CH_LST_SR")
    path = tmp_path / "current_missing_parton.root"
    with uproot.recreate(path) as output:
        for category in layout.categories:
            tree = output.mktree(category.base_sr_category, {"tllq": "float64"})
            tree.extend({"tllq": np.zeros(category.public_array_length)})
    assert infer_legacy_missing_parton_layout(path).registry == "ALL_CH_LST_SR"
    assert len(validate_legacy_missing_parton_payload(path)["2los_onZ_1tau"]) == 6


def _payload(channels=(TARGET, OTHER), *, background=True, signal=True):
    output = {}
    for channel in channels:
        if background:
            for family in FAMILIES:
                key = nominal_schema.scalar_nominal_key(family)
                companion = nominal_schema.sumw2_key(family)
                scalar = _scalar("background", 2.0, family=family, channel=channel)
                sumw2 = _scalar("background", 4.0, family=family, companion=True, channel=channel)
                if key in output:
                    output[key] += scalar
                    output[companion] += sumw2
                else:
                    output[key] = scalar
                    output[companion] = sumw2
        if signal:
            other_key = nominal_schema.scalar_nominal_key("ptz_wtau")
            other_companion = nominal_schema.sumw2_key("ptz_wtau")
            other_scalar = _scalar("signal", 2.0, family="ptz_wtau", channel=channel)
            other_sumw2 = _scalar("signal", 4.0, family="ptz_wtau", companion=True, channel=channel)
            if other_key in output:
                output[other_key] += other_scalar
                output[other_companion] += other_sumw2
            else:
                output[other_key] = other_scalar
                output[other_companion] = other_sumw2
            key = nominal_schema.eft_nominal_key("njets")
            eft = _eft("signal", 3.0, channel=channel)
            sumw2 = _scalar("signal", 9.0, family="njets", companion=True, channel=channel)
            if key in output:
                output[key] += eft
            else:
                output[key] = eft
            companion = nominal_schema.sumw2_key("njets")
            if companion in output:
                output[companion] += sumw2
            else:
                output[companion] = sumw2
    for histogram in output.values():
        dense_name = (
            histogram.dense_axis.name if isinstance(histogram, HistEFT)
            else histogram.dense_axes[0].name
        )
        for coordinate in tuple(histogram.categorical_keys):
            for systematic, weight in (("shapeUp", 1.2), ("shapeDown", 0.8)):
                fill_kwargs = {
                    "process": coordinate.process,
                    "channel": coordinate.channel,
                    "systematic": systematic,
                    "appl": coordinate.appl,
                    dense_name: np.asarray([0.5]),
                    "weight": np.asarray([weight]),
                }
                if isinstance(histogram, HistEFT):
                    fill_kwargs["eft_coeff"] = np.asarray([[1.25, 2.0, 3.0]])
                histogram.fill(**fill_kwargs)
    return output


def _write_fixture(path, *, background=True, signal=True):
    payload = _payload(background=background, signal=signal)
    _write_versioned(path, payload, _policy_for_families(FAMILIES))
    return payload


def _assert_histograms_equal(first, second):
    assert set(first) == set(second)
    for key in first:
        left = first[key]
        right = second[key]
        assert type(left) is type(right)
        assert left.axes.name == right.axes.name
        assert list(left.axes["process"]) == list(right.axes["process"])
        assert list(left.axes["channel"]) == list(right.axes["channel"])
        assert list(left.axes["systematic"]) == list(right.axes["systematic"])
        left_dense = left.dense_axis if isinstance(left, HistEFT) else left.dense_axes[0]
        right_dense = right.dense_axis if isinstance(right, HistEFT) else right.dense_axes[0]
        assert left_dense.name == right_dense.name
        assert np.array_equal(left_dense.edges, right_dense.edges)
        assert left_dense.traits.underflow == right_dense.traits.underflow
        assert left_dense.traits.overflow == right_dense.traits.overflow
        assert set(left.view(flow=True, as_dict=True)) == set(right.view(flow=True, as_dict=True))
        for coordinate, values in left.view(flow=True, as_dict=True).items():
            np.testing.assert_allclose(values, right.view(flow=True, as_dict=True)[coordinate])


def test_selective_path_retains_only_declared_closure_and_channels(tmp_path, monkeypatch):
    path = tmp_path / "one.pkl.gz"
    _write_fixture(path)
    eager, _ = datacard_tools.load_and_merge_histogram_pkls([str(path)])
    expected = {
        key: histogram.prune("channel", [TARGET])
        for key, histogram in eager.items()
        if key.startswith("njets")
    }
    monkeypatch.setattr(
        datacard_tools, "get_hist_from_pkl",
        lambda *_args, **_kwargs: pytest.fail("eager loader called on selective path"),
    )
    selected, report = datacard_tools.load_and_merge_histogram_pkls(
        [str(path)], selected_families=["njets"], channel_patterns=[r"^3l_onZ_2b"],
    )
    assert set(selected) == {
        "njets__scalar_nominal", "njets__eft_nominal", "njets_sumw2"
    }
    assert set(report["files"][0]["discarded_keys"]) == {
        "ptz_wtau__scalar_nominal", "ptz_wtau_sumw2"
    }
    assert report["files"][0]["resolved_channels"] == [TARGET]
    assert report["validation_scope"] == "artifact_identity_and_selected_content_closure"
    _assert_histograms_equal(selected, expected)
    eager_card = datacard_tools.DatacardMaker(
        hists=eager, var_lst=["njets"], channel_patterns=[r"^3l_onZ_2b"],
        out_dir=str(tmp_path), do_nuisance=False, verbose=False,
    )
    selective_card = datacard_tools.DatacardMaker(
        hists=selected, var_lst=["njets"], channel_patterns=[r"^3l_onZ_2b"],
        out_dir=str(tmp_path), do_nuisance=False, verbose=False,
    )
    _assert_histograms_equal(
        {key: eager_card.hists[key] for key in ("njets", "njets_sumw2")},
        {key: selective_card.hists[key] for key in ("njets", "njets_sumw2")},
    )


def test_selective_fails_on_missing_companion_and_bad_sidecar(tmp_path):
    path = tmp_path / "one.pkl.gz"
    _write_fixture(path)
    metadata_path = metadata_sidecar_path(path)
    metadata = json.loads(metadata_path.read_text())
    metadata["artifact"]["nominal_container_layout"] = "wrong"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(Exception, match="layout is incompatible"):
        datacard_tools.load_and_merge_histogram_pkls(
            [str(path)], selected_families=["njets"], channel_patterns=[TARGET],
        )


def test_selective_missing_sumw2_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "one.pkl.gz"
    _write_fixture(path)
    original_iterator = datacard_tools.iterate_hist_from_pkl

    def missing_companion(*args, **kwargs):
        return (
            (key, value) for key, value in original_iterator(*args, **kwargs)
            if key != "njets_sumw2"
        )

    monkeypatch.setattr(datacard_tools, "iterate_hist_from_pkl", missing_companion)
    with pytest.raises(Exception, match="closure incomplete.*njets_sumw2"):
        datacard_tools.load_and_merge_histogram_pkls(
            [str(path)], selected_families=["njets"], channel_patterns=[TARGET],
        )


@pytest.mark.parametrize(
    "change, expected_error",
    [
        ("identity", "identity mismatch"),
        ("process_content", "Selected-content process mismatch"),
    ],
)
def test_selective_rejects_identity_or_process_contract_mismatch(
    tmp_path, change, expected_error,
):
    path = tmp_path / "one.pkl.gz"
    _write_fixture(path)
    metadata_path = metadata_sidecar_path(path)
    metadata = json.loads(metadata_path.read_text())
    if change == "identity":
        metadata["artifact"]["pkl_sha256"] = "0" * 64
    else:
        metadata["sumw2_content_manifest"]["families"]["njets"][
            "eft_nominal_processes"
        ] = ["different_process"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(Exception, match=expected_error):
        datacard_tools.load_and_merge_histogram_pkls(
            [str(path)], selected_families=["njets"], channel_patterns=[TARGET],
        )


def test_make_cards_passes_existing_selectors_to_loader(monkeypatch, tmp_path):
    captured = {}
    def fake_loader(_paths, **kwargs):
        captured.update(kwargs)
        return {}, {"schema": "split_sibling_v1"}
    monkeypatch.setattr(make_cards.datacard_tools, "load_and_merge_histogram_pkls", fake_loader)
    monkeypatch.setattr(make_cards, "_emit_merge_report", lambda *_args: None)
    args = make_cards.build_arg_parser().parse_args([
        "input.pkl.gz", "--merge-only", "--out-dir", str(tmp_path),
        "--var-lst", "njets", "--ch-lst", r"^3l_onZ_2b",
    ])
    make_cards._run_producer(args, ["input.pkl.gz"])
    assert captured["selected_families"] == ["njets"]
    assert captured["channel_patterns"] == [r"^3l_onZ_2b"]


def test_selected_cache_cannot_claim_a_complete_artifact(tmp_path):
    args = make_cards.build_arg_parser().parse_args([
        "input.pkl.gz", "--merge-only", "--out-dir", str(tmp_path),
        "--var-lst", "njets", "--cache-merged-pkl", "cache.pkl.gz",
    ])
    with pytest.raises(ValueError, match="cannot be cached"):
        make_cards._run_producer(args, ["input.pkl.gz"])
    assert not (tmp_path / "cache.pkl.gz").exists()


def test_multi_pkl_selected_merge_matches_eager_for_target(tmp_path):
    first = tmp_path / "background.pkl.gz"
    second = tmp_path / "signal.pkl.gz"
    _write_fixture(first, signal=False)
    _write_fixture(second, background=False)
    paths = [str(first), str(second)]
    eager, _ = datacard_tools.load_and_merge_histogram_pkls(paths)
    selected, report = datacard_tools.load_and_merge_histogram_pkls(
        paths, selected_families=["njets"], channel_patterns=[r"^3l_onZ_2b"],
    )
    expected = {
        key: histogram.prune("channel", [TARGET])
        for key, histogram in eager.items() if key.startswith("njets")
    }
    _assert_histograms_equal(selected, expected)
    assert all("ptz_wtau__scalar_nominal" in item["discarded_keys"] for item in report["files"] if "background" in item["path"])


def test_multi_pkl_family_absent_by_sidecar_scope_is_not_required(tmp_path):
    first = tmp_path / "full.pkl.gz"
    second = tmp_path / "other_family_only.pkl.gz"
    _write_fixture(first)
    other_payload = {
        key: value for key, value in _payload(background=True, signal=False).items()
        if key.startswith("ptz_wtau")
    }
    _write_versioned(
        second, other_payload, _policy_for_families(("ptz_wtau",)),
    )
    selected, report = datacard_tools.load_and_merge_histogram_pkls(
        [str(first), str(second)], selected_families=["njets"],
        channel_patterns=[r"^3l_onZ_2b"],
    )
    assert set(selected) == {
        "njets__scalar_nominal", "njets__eft_nominal", "njets_sumw2"
    }
    assert report["files"][1]["retained_keys"] == []
