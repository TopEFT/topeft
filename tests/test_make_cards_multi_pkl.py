from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import hist
import pytest

from analysis.topeft_run2 import make_cards
from topcoffea.modules.sparseHist import SparseHist
from topeft.modules import datacard_tools
from topeft.modules import histogram_artifact
from topeft.modules import nominal_schema
from topeft.modules.sumw2_policy import SUMW2_PROVENANCE_SCHEMA_VERSION


def _make_hist(processes, dense_name, nbins=1, dense_hi=None, channel="ch1"):
    if dense_hi is None:
        dense_hi = float(nbins)
    fill_value = float(dense_hi) / (2.0 * float(nbins))
    h = SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.Regular(nbins, 0.0, float(dense_hi), name=dense_name),
        storage="Double",
    )
    for proc in processes:
        h.fill(process=proc, channel=channel, **{dense_name: fill_value}, weight=1.0)
    return h


def _make_hist_with_systematic(process, dense_name, channel, systematic):
    h = SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name=dense_name),
        storage="Double",
    )
    h.fill(
        process=process,
        channel=channel,
        systematic=systematic,
        **{dense_name: 0.5},
        weight=1.0,
    )
    return h


def _make_payload(
    key,
    processes,
    *,
    with_sumw2=True,
    nbins=1,
    dense_hi=None,
    channel="ch1",
):
    payload = {
        key: _make_hist(
            processes, key, nbins=nbins, dense_hi=dense_hi, channel=channel
        )
    }
    if with_sumw2:
        sumw2_key = f"{key}_sumw2"
        payload[sumw2_key] = _make_hist(
            processes,
            sumw2_key,
            nbins=nbins,
            dense_hi=dense_hi,
            channel=channel,
        )
    return payload


def _make_schema_payload(family, process, channel):
    return {
        nominal_schema.scalar_nominal_key(family): _make_hist(
            [process], family, channel=channel
        ),
        nominal_schema.sumw2_key(family): _make_hist(
            [process], nominal_schema.sumw2_key(family), channel=channel
        ),
    }


def _make_provenance(families, *, dataset, process, warning=""):
    return {
        "schema_version": SUMW2_PROVENANCE_SCHEMA_VERSION,
        "source": "explicit",
        "requested_mode": "full_diagnostics",
        "resolved_mode": "full_diagnostics",
        "signal_sample_profile": "unrestricted",
        "normalized_rules": [],
        "runtime_histogram_families": list(families),
        "resolved_datasets": [dataset],
        "resolved_processes": [process],
        "resolved_targets": [
            {"dataset": dataset, "process": process, "family": family}
            for family in families
        ],
        "warnings": [warning] if warning else [],
    }


def test_merge_histogram_pkls_succeeds_for_disjoint_processes(monkeypatch):
    payloads = {
        "a.pkl.gz": _make_payload("met", ["proc_a"], channel="channel_a"),
        "b.pkl.gz": _make_payload("met", ["proc_b"], channel="channel_b"),
    }

    def fake_loader(path, allow_empty=False):
        assert allow_empty is False
        return payloads[path]

    monkeypatch.setattr(datacard_tools, "get_hist_from_pkl", fake_loader)

    merged, report = datacard_tools.load_and_merge_histogram_pkls(
        ["a.pkl.gz", "b.pkl.gz"],
    )

    assert set(merged["met"].axes["process"]) == {"proc_a", "proc_b"}
    assert report["contribution_identity"].startswith("payload_component_key")


def test_ptz_and_ptll_remain_distinct_multi_pkl_family_identities(monkeypatch):
    category = "3l_m_offZ_low_1b_2j"
    payloads = {
        "historical.pkl.gz": _make_payload(
            "ptz", ["shared_proc"], channel=category
        ),
        "canonical.pkl.gz": _make_payload(
            "ptll", ["shared_proc"], channel=category
        ),
    }
    monkeypatch.setattr(
        datacard_tools,
        "get_hist_from_pkl",
        lambda path, allow_empty=False: payloads[path],
    )

    merged, report = datacard_tools.load_and_merge_histogram_pkls(
        ["historical.pkl.gz", "canonical.pkl.gz"]
    )

    assert set(merged) == {"ptz", "ptz_sumw2", "ptll", "ptll_sumw2"}
    assert report["contribution_identity"].startswith("payload_component_key")
    assert "ptll" not in payloads["historical.pkl.gz"]
    assert "ptz" not in payloads["canonical.pkl.gz"]


def test_merge_histogram_pkls_fails_when_sumw2_missing(monkeypatch):
    payloads = {
        "broken.pkl.gz": _make_payload("met", ["proc_a"], with_sumw2=False),
    }

    monkeypatch.setattr(datacard_tools, "get_hist_from_pkl", lambda path, allow_empty=False: payloads[path])

    with pytest.raises(RuntimeError, match="missing required \\*_sumw2 companions"):
        datacard_tools.load_and_merge_histogram_pkls(["broken.pkl.gz"])


def test_merge_histogram_pkls_fails_on_dense_axis_edges_mismatch(monkeypatch):
    payloads = {
        "a.pkl.gz": _make_payload(
            "met", ["proc_a"], dense_hi=1.0, channel="channel_a"
        ),
        "b.pkl.gz": _make_payload(
            "met", ["proc_b"], dense_hi=2.0, channel="channel_b"
        ),
    }

    monkeypatch.setattr(datacard_tools, "get_hist_from_pkl", lambda path, allow_empty=False: payloads[path])

    with pytest.raises(ValueError, match="Dense-axis edges mismatch"):
        datacard_tools.load_and_merge_histogram_pkls(
            ["a.pkl.gz", "b.pkl.gz"],
        )


def test_process_overlap_with_disjoint_channels_is_transparent(monkeypatch):
    payloads = {
        "a.pkl.gz": _make_payload(
            "met", ["shared_proc"], channel="channel_a"
        ),
        "b.pkl.gz": _make_payload(
            "met", ["shared_proc"], channel="channel_b"
        ),
    }

    monkeypatch.setattr(datacard_tools, "get_hist_from_pkl", lambda path, allow_empty=False: payloads[path])

    merged, report = datacard_tools.load_and_merge_histogram_pkls(
        ["a.pkl.gz", "b.pkl.gz"],
    )
    assert "met" in merged
    assert set(merged["met"].axes["channel"]) == {"channel_a", "channel_b"}
    assert report["num_year_coverage_mismatches"] == 0


def test_genuine_duplicate_contribution_is_rejected(monkeypatch):
    payloads = {
        "a.pkl.gz": {
            "met": _make_hist(["shared_proc"], "met", channel="channel_a"),
            "met_sumw2": _make_hist(
                ["shared_proc"], "met_sumw2", channel="channel_a"
            ),
        },
        "b.pkl.gz": {
            "met": _make_hist(["shared_proc"], "met", channel="channel_a"),
            "met_sumw2": _make_hist(
                ["shared_proc"], "met_sumw2", channel="channel_a"
            ),
        },
    }
    monkeypatch.setattr(
        datacard_tools,
        "get_hist_from_pkl",
        lambda path, allow_empty=False: payloads[path],
    )

    with pytest.raises(RuntimeError, match="Duplicate histogram contribution support"):
        datacard_tools.load_and_merge_histogram_pkls(
            ["a.pkl.gz", "b.pkl.gz"],
        )


def test_repeated_channel_with_distinct_process_is_supported(monkeypatch):
    payloads = {
        "a.pkl.gz": _make_payload("met", ["proc_a"], channel="3l_onZ_2b_4j"),
        "b.pkl.gz": _make_payload("met", ["proc_b"], channel="3l_onZ_2b_4j"),
    }
    monkeypatch.setattr(
        datacard_tools,
        "get_hist_from_pkl",
        lambda path, allow_empty=False: payloads[path],
    )

    merged, _report = datacard_tools.load_and_merge_histogram_pkls(
        ["a.pkl.gz", "b.pkl.gz"],
    )
    assert set(merged["met"].axes["process"]) == {"proc_a", "proc_b"}


def test_complete_categorical_identity_distinguishes_systematic_fragments(monkeypatch):
    payloads = {
        "nominal.pkl.gz": {
            "met": _make_hist_with_systematic(
                "sharedUL18", "met", "shared_channel", "nominal"
            ),
            "met_sumw2": _make_hist_with_systematic(
                "sharedUL18", "met_sumw2", "shared_channel", "nominal"
            ),
        },
        "variation.pkl.gz": {
            "met": _make_hist_with_systematic(
                "sharedUL18", "met", "shared_channel", "scaleUp"
            ),
            "met_sumw2": _make_hist_with_systematic(
                "sharedUL18", "met_sumw2", "shared_channel", "scaleUp"
            ),
        },
    }
    monkeypatch.setattr(
        datacard_tools,
        "get_hist_from_pkl",
        lambda path, allow_empty=False: payloads[path],
    )

    merged, _report = datacard_tools.load_and_merge_histogram_pkls(
        ["nominal.pkl.gz", "variation.pkl.gz"]
    )

    assert set(merged["met"].axes["systematic"]) == {"nominal", "scaleUp"}


def test_legacy_cross_run_histogram_composition_is_rejected(monkeypatch):
    payloads = {
        "run2.pkl.gz": _make_payload(
            "met", ["backgroundUL18"], channel="2lss_m_4j"
        ),
        "run3.pkl.gz": _make_payload(
            "met", ["background2022"], channel="3l_onZ_2b_4j"
        ),
    }
    monkeypatch.setattr(
        datacard_tools,
        "get_hist_from_pkl",
        lambda path, allow_empty=False: payloads[path],
    )

    with pytest.raises(
        RuntimeError, match=r"Run 2 \+ Run 3 composition is unsupported"
    ):
        datacard_tools.load_and_merge_histogram_pkls(
            ["run2.pkl.gz", "run3.pkl.gz"]
        )


def test_single_input_with_cross_run_contributions_is_rejected(monkeypatch):
    payload = _make_payload(
        "met",
        ["backgroundUL18", "background2022"],
        channel="3l_onZ_2b_4j",
    )
    monkeypatch.setattr(
        datacard_tools,
        "get_hist_from_pkl",
        lambda path, allow_empty=False: payload,
    )

    with pytest.raises(
        RuntimeError, match=r"Run 2 \+ Run 3 composition is unsupported"
    ):
        datacard_tools.load_and_merge_histogram_pkls(["mixed.pkl.gz"])


def test_split_family_provenance_composes_first_occurrence_ordered_union():
    mixed = ("njets", "lj0pt", "ptz", "ptll", "ptz_wtau", "lt")
    sibling = ("njets", "lj0pt", "ptz", "ptll", "lt")
    composed = histogram_artifact._compose_sumw2_storage_provenance(
        (
            _make_provenance(mixed, dataset="shared_dataset", process="shared_proc"),
            _make_provenance(
                sibling, dataset="shared_dataset", process="shared_proc"
            ),
        )
    )

    assert composed["runtime_histogram_families"] == list(mixed)
    assert [target["family"] for target in composed["resolved_targets"]] == [
        "njets",
        "lj0pt",
        "ptz",
        "ptll",
        "ptz_wtau",
        "lt",
    ]
    assert histogram_artifact._ordered_family_union((mixed, sibling, mixed)) == list(mixed)


def test_split_family_provenance_preserves_policy_and_allocation_guards():
    first = _make_provenance(("njets",), dataset="dataset_a", process="process_a")
    incompatible = _make_provenance(
        ("njets",), dataset="dataset_b", process="process_b", warning="changed"
    )
    with pytest.raises(RuntimeError, match="policy-control field 'warnings'"):
        histogram_artifact._compose_sumw2_storage_provenance((first, incompatible))

    different_dataset = _make_provenance(
        ("njets",), dataset="dataset_b", process="process_a"
    )
    different_process = _make_provenance(
        ("njets",), dataset="dataset_a", process="process_b"
    )
    composed = histogram_artifact._compose_sumw2_storage_provenance(
        (first, different_dataset, different_process)
    )
    assert composed["resolved_datasets"] == ["dataset_a", "dataset_b"]
    assert composed["resolved_processes"] == ["process_a", "process_b"]


def test_sidecar_merge_composes_partial_content_manifests(monkeypatch):
    mixed = ("njets", "ptz_wtau")
    sibling = ("njets",)
    first_provenance = _make_provenance(
        mixed, dataset="shared_dataset", process="shared_proc"
    )
    second_provenance = _make_provenance(
        sibling, dataset="shared_dataset", process="shared_proc"
    )
    composed = histogram_artifact._compose_sumw2_storage_provenance(
        (first_provenance, second_provenance)
    )

    def sidecar(provenance):
        return {
            "artifact": {
                "artifact_kind": "processor_output",
                "nominal_container_schema_version": 2,
                "nominal_container_layout": "split_sibling_v1",
                "pkl_basename": f"{provenance['resolved_datasets'][0]}.pkl.gz",
                "pkl_sha256": provenance["resolved_datasets"][0],
            },
            "sumw2_storage_provenance": provenance,
            "sumw2_content_manifest": {
                "families": {
                    family: {
                        "dimensionality": 1,
                        "required_sumw2_processes": [provenance["resolved_processes"][0]],
                    }
                    for family in provenance["runtime_histogram_families"]
                }
            },
        }

    monkeypatch.setattr(
        histogram_artifact,
        "_compose_merged_contract_set",
        lambda sidecars: (composed, None, None, None),
    )
    report = histogram_artifact.merge_histogram_sidecars(
        (sidecar(first_provenance), sidecar(second_provenance))
    )

    assert report["sumw2_storage_provenance"]["runtime_histogram_families"] == list(
        mixed
    )
    assert sorted(report["required_sumw2_processes"]) == list(mixed)


def test_merge_nominal_mappings_validates_partial_inputs_and_complete_output():
    families = ("njets", "ptz_wtau")
    first = _make_schema_payload("njets", "shared_proc", "channel_a")
    second = _make_schema_payload("ptz_wtau", "shared_proc", "channel_b")

    merged = nominal_schema.merge_nominal_mappings(
        (first, second), runtime_families=families
    )

    assert tuple(merged) == (
        nominal_schema.scalar_nominal_key("njets"),
        nominal_schema.sumw2_key("njets"),
        nominal_schema.scalar_nominal_key("ptz_wtau"),
        nominal_schema.sumw2_key("ptz_wtau"),
    )
    nominal_schema.validate_nominal_mapping(
        merged, runtime_families=families
    )


def test_partial_input_validation_rejects_claimed_missing_or_malformed_family():
    missing_nominal = {
        nominal_schema.sumw2_key("njets"): _make_hist(
            ["proc"], nominal_schema.sumw2_key("njets")
        )
    }
    with pytest.raises(ValueError, match="orphan statistical companion"):
        nominal_schema.merge_nominal_mappings(
            (missing_nominal,), runtime_families=("njets", "ptz_wtau")
        )

    malformed = {
        nominal_schema.scalar_nominal_key("njets"): object(),
    }
    with pytest.raises(TypeError, match="must be an exact SparseHist"):
        nominal_schema.merge_nominal_mappings(
            (malformed,), runtime_families=("njets", "ptz_wtau")
        )


def test_make_cards_parser_accepts_multiple_pkls():
    parser = make_cards.build_arg_parser()
    args = parser.parse_args(
        [
            "a.pkl.gz",
            "b.pkl.gz",
            "--year-coverage-policy",
            "error",
            "--merge-only",
        ]
    )

    assert args.pkl_file == ["a.pkl.gz", "b.pkl.gz"]
    assert args.year_coverage_policy == "error"
    assert args.merge_only is True


def test_make_cards_parser_default_year_coverage_policy_is_warn():
    parser = make_cards.build_arg_parser()
    args = parser.parse_args(["a.pkl.gz"])

    assert args.year_coverage_policy == "warn"


@pytest.mark.parametrize("year_coverage_policy", ("warn", "error", "off"))
def test_make_cards_parser_accepts_explicit_year_coverage_policy(
    year_coverage_policy,
):
    parser = make_cards.build_arg_parser()
    args = parser.parse_args(
        ["a.pkl.gz", "--year-coverage-policy", year_coverage_policy]
    )

    assert args.year_coverage_policy == year_coverage_policy


@pytest.mark.parametrize(
    "cli_args, expected_policy",
    [([], "warn"), (["--year-coverage-policy", "off"], "off")],
)
def test_make_cards_main_propagates_resolved_year_coverage_policy(
    monkeypatch,
    cli_args,
    expected_policy,
):
    captured = {}

    def _fake_load(pkl_paths, **kwargs):
        captured["pkl_paths"] = pkl_paths
        captured.update(kwargs)
        return {}, {"num_inputs": len(pkl_paths)}

    monkeypatch.setattr(make_cards, "load_and_merge_histogram_pkls", _fake_load)
    monkeypatch.setattr(
        sys,
        "argv",
        ["make_cards.py", "input.pkl.gz", "--merge-only", *cli_args],
    )

    make_cards.main()

    assert captured == {
        "pkl_paths": ["input.pkl.gz"],
        "require_sumw2": True,
        "year_coverage_policy": expected_policy,
    }


@pytest.mark.parametrize("year_coverage_policy", ("warn", "error", "off"))
def test_condor_options_propagate_resolved_year_coverage_policy(year_coverage_policy):
    dc = SimpleNamespace(
        do_mc_stat=False,
        verbose=False,
        use_real_data=False,
        do_nuisance=False,
        year_lst=[],
        drop_syst=[],
        sr_registry="an_v9",
        skip_missing_parton_rate_syst=False,
        binning_mode="fitting",
    )

    options = make_cards._build_condor_base_other_opts(dc, year_coverage_policy)

    assert options[-2:] == ["--year-coverage-policy", year_coverage_policy]


def test_condor_options_propagate_explicit_rate_systematics_override():
    dc = SimpleNamespace(
        do_mc_stat=False,
        verbose=False,
        use_real_data=False,
        do_nuisance=False,
        year_lst=[],
        drop_syst=[],
        sr_registry="an_v9",
        skip_missing_parton_rate_syst=False,
        binning_mode="fitting",
    )

    options = make_cards._build_condor_base_other_opts(
        dc,
        "warn",
        rate_syst_json="custom/rate_systematics.json",
    )

    assert options[-6:] == [
        "--rate-syst-json",
        "custom/rate_systematics.json",
        "--binning",
        "fitting",
        "--year-coverage-policy",
        "warn",
    ]


def test_use_selected_materializes_signal_only_output_without_mutating_source(
    monkeypatch,
    tmp_path,
):
    source_path = tmp_path / "input_selected_wcs.json"
    source_content = {"signal": ["ctW"], "background": []}
    source_path.write_text(json.dumps(source_content), encoding="utf-8")
    output_dir = tmp_path / "cards"

    class fake_datacard_maker:
        hists = {"met": object()}
        scalings = []

        def __init__(self, **kwargs):
            pass

        def is_signal(self, process):
            return process == "signal"

        def processes(self, _distribution):
            return ["signal", "background"]

    captured_selected_wcs = {}

    def capture_run_local(_dc, _dists, _channels, selected_wcs, *_args):
        captured_selected_wcs.update(selected_wcs)

    monkeypatch.setattr(
        make_cards,
        "load_and_merge_histogram_pkls",
        lambda *args, **kwargs: ({}, {}),
    )
    monkeypatch.setattr(make_cards, "_emit_merge_report", lambda *args: None)
    monkeypatch.setattr(make_cards, "DatacardMaker", fake_datacard_maker)
    monkeypatch.setattr(make_cards, "run_local", capture_run_local)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_cards.py",
            "input.pkl.gz",
            "--out-dir",
            str(output_dir),
            "--var-lst",
            "met",
            "--use-selected",
            str(source_path),
        ],
    )

    make_cards.main()

    assert json.loads(source_path.read_text(encoding="utf-8")) == source_content
    assert json.loads(
        (output_dir / "selectedWCs.txt").read_text(encoding="utf-8")
    ) == {"signal": ["ctW"]}
    assert captured_selected_wcs == {"signal": ["ctW"], "background": []}


def test_use_selected_same_file_is_reused_only_when_canonical(tmp_path):
    output_path = tmp_path / "selectedWCs.txt"
    selected_wcs = {"signal": ["ctW"]}
    output_path.write_text(json.dumps(selected_wcs), encoding="utf-8")

    resolved_path = make_cards._materialize_selected_wcs(
        str(tmp_path),
        selected_wcs,
        source_path=str(output_path),
    )

    assert resolved_path == str(output_path)
    assert json.loads(output_path.read_text(encoding="utf-8")) == selected_wcs

    noncanonical_selected_wcs = {"signal": ["ctW"], "background": []}
    output_path.write_text(
        json.dumps(noncanonical_selected_wcs),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical signal-only representation"):
        make_cards._materialize_selected_wcs(
            str(tmp_path),
            selected_wcs,
            source_path=str(output_path),
        )
    assert json.loads(output_path.read_text(encoding="utf-8")) == noncanonical_selected_wcs


def test_resolve_pkl_paths_from_file(tmp_path):
    pkl_list = tmp_path / "pkls.txt"
    pkl_list.write_text(
        "\n".join(
            [
                "# comment line",
                "",
                "/tmp/a.pkl.gz",
                "/tmp/b.pkl.gz",
            ]
        )
        + "\n"
    )

    parser = make_cards.build_arg_parser()
    args = parser.parse_args(["--pkl-list-file", str(pkl_list)])
    resolved = make_cards._resolve_pkl_paths(args, parser)

    assert resolved == ["/tmp/a.pkl.gz", "/tmp/b.pkl.gz"]
