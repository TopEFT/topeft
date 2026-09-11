import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cloudpickle
import hist
import numpy as np
import pytest

from topcoffea.modules.sparseHist import SparseHist


def _load_run_data_driven_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "analysis" / "topeft_run2" / "run_data_driven.py"
    spec = importlib.util.spec_from_file_location("run_data_driven", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


run_data_driven = _load_run_data_driven_module()


def _legacy_context_inputs(tmp_path):
    source_path = tmp_path / "canonical_source.pkl.gz"
    state_path = tmp_path / "campaign_state.json"
    source_command = [
        "python",
        "run_analysis.py",
        "--sr",
        "-y",
        "2018",
        "--hist-vars",
        "lt",
        "--category-groups",
        "3l_onZ_tau",
        "--all-analysis",
    ]
    state = {
        "production_profile": "t0_sr_statonly",
        "topeft_git_commit": "1" * 40,
        "region": "SR",
        "blocks": [
            {
                "id": "legacy_block",
                "source_status": "ready",
                "expected_nominal_path": str(source_path),
                "category_groups": ["3l_onZ_tau"],
                "histograms": ["lt"],
                "years": ["2018"],
                "source_command_argv": source_command,
            }
        ],
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    input_sidecar = {
        "artifact": {
            "artifact_kind": "processor_output",
            "pkl_size_bytes": 123,
            "pkl_sha256": "2" * 64,
        },
        "sumw2_storage_provenance": {},
    }
    return state_path, source_path, state, input_sidecar


def _legacy_category_config(*, lepton_channel="3l_onZ_1b"):
    return {
        "ALL_CH_LST_SR": {
            "3l_onZ_tau": {
                "lep_chan_lst": [[lepton_channel]],
                "lep_flav_lst": ["eee"],
                "appl_lst": ["isSR_3l"],
                "jet_lst": ["=2"],
            },
            "4l": {
                "lep_chan_lst": [["4l"]],
                "lep_flav_lst": ["eeee"],
                "appl_lst": ["isSR_4l"],
                "jet_lst": ["=2"],
            },
        }
    }


def _install_legacy_context_stubs(
    monkeypatch,
    input_sidecar,
    *,
    families=("lt",),
    category_config=None,
    producer_source=None,
):
    monkeypatch.setattr(
        run_data_driven,
        "read_histogram_sidecar",
        lambda _path: input_sidecar,
    )
    monkeypatch.setattr(
        run_data_driven,
        "resolved_policy_from_provenance",
        lambda _provenance: SimpleNamespace(runtime_histogram_families=families),
    )
    if producer_source is None:
        producer_source = Path(run_data_driven.analysis_processor.__file__).read_text(
            encoding="utf-8"
        )
    if category_config is None:
        category_config = _legacy_category_config()
    monkeypatch.setattr(
        run_data_driven.subprocess,
        "run",
        lambda command, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=(
                json.dumps(category_config)
                if command[-1].endswith(":topeft/channels/ch_lst.json")
                else producer_source
            ),
        ),
    )


class FakeHist:
    def __init__(self, processes):
        self._processes = list(processes)
        self.axes = {"process": tuple(self._processes)}

    def remove(self, axis_name, labels):
        assert axis_name == "process"
        keep = [p for p in self._processes if p not in set(labels)]
        return FakeHist(keep)


class DummyProducer:
    output_hist = {}
    calls = []
    get_calls = 0
    iter_calls = 0

    def __init__(self, inputHist, outputName, iterator_mode=False):
        self.inputHist = inputHist
        self.outputName = outputName
        self.iterator_mode = iterator_mode
        DummyProducer.calls.append((inputHist, outputName, iterator_mode))

    def getDataDrivenHistogram(self):
        DummyProducer.get_calls += 1
        return DummyProducer.output_hist

    def iter_data_driven_histograms(self):
        DummyProducer.iter_calls += 1
        yield from DummyProducer.output_hist.items()


@pytest.fixture(autouse=True)
def clear_dummy_state():
    DummyProducer.calls.clear()
    DummyProducer.output_hist = {}
    DummyProducer.get_calls = 0
    DummyProducer.iter_calls = 0
    yield
    DummyProducer.calls.clear()
    DummyProducer.output_hist = {}
    DummyProducer.get_calls = 0
    DummyProducer.iter_calls = 0


def _load_pkl(pkl_path: Path):
    with gzip.open(pkl_path, "rb") as stream:
        return cloudpickle.load(stream)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _build_data_driven_input_hist():
    return SparseHist(
        hist.axis.StrCategory([], name="process", growth=True),
        hist.axis.StrCategory([], name="channel", growth=True),
        hist.axis.StrCategory([], name="systematic", growth=True),
        hist.axis.StrCategory([], name="appl", growth=True),
        hist.axis.Regular(1, 0.0, 1.0, name="met"),
        storage="Double",
    )


def _fill_data_driven_histogram(entries):
    histo = _build_data_driven_input_hist()
    for entry in entries:
        histo.fill(
            process=entry["process"],
            channel=entry["channel"],
            systematic=entry.get("systematic", "nominal"),
            appl=entry["appl"],
            met=np.array([entry.get("met", 0.5)], dtype=float),
            weight=np.array([entry["weight"]], dtype=float),
        )
    return histo


def _write_histograms(path: Path, payload):
    with gzip.open(path, "wb") as stream:
        cloudpickle.dump(payload, stream)


def _run_histograms(tmp_path, histograms, *extra_args):
    input_path = tmp_path / "input.pkl.gz"
    output_path = tmp_path / "output.pkl.gz"
    _write_histograms(input_path, histograms)
    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--quiet",
            *extra_args,
        ]
    )
    return output_path


def _run_with_dd_report(tmp_path, histograms, *extra_args):
    return _run_histograms(tmp_path, histograms, "--dd-report", *extra_args)


def _single_bin_total(histo, process_name):
    values = histo.integrate("process", [process_name]).integrate("systematic", "nominal").values(
        flow=True
    )[()]
    return float(np.asarray(values).sum())


def test_run_data_driven_from_pkl_paths(tmp_path, monkeypatch):
    input_path = tmp_path / "input.pkl.gz"
    _write_histograms(input_path, {"seed": FakeHist(["dataUL17"])})
    output_path = tmp_path / "output.pkl.gz"

    DummyProducer.output_hist = {"njets": FakeHist(["flipsUL17"])}
    monkeypatch.setattr(run_data_driven, "DataDrivenProducer", DummyProducer)

    run_data_driven.main(
        ["--input-pkl", str(input_path), "--output-pkl", str(output_path)]
    )

    assert DummyProducer.calls == [(str(input_path), str(output_path), True)]
    assert DummyProducer.iter_calls == 1
    assert DummyProducer.get_calls == 0
    result = _load_pkl(output_path)
    assert list(result["njets"].axes["process"]) == ["flipsUL17"]


def test_run_data_driven_rejects_deprecated_envelope_before_input_validation(tmp_path, monkeypatch):
    input_path = tmp_path / "input.pkl.gz"
    _write_histograms(input_path, {"seed": FakeHist(["dataUL18"])})
    output_path = tmp_path / "output.pkl.gz"

    def fail_if_validated(_path):
        raise AssertionError("input validation must not run")

    monkeypatch.setattr(run_data_driven, "_validate_input_path", fail_if_validated)

    with pytest.raises(RuntimeError, match="combined renorm/fact envelope"):
        run_data_driven.main(
            [
                "--input-pkl",
                str(input_path),
                "--output-pkl",
                str(output_path),
                "--only-flips",
                "--apply-renormfact-envelope",
            ]
        )

    assert not output_path.exists()
    assert DummyProducer.calls == []


def test_run_data_driven_rejects_manual_metadata_sidecar_option():
    with pytest.raises(SystemExit):
        run_data_driven.main(["--metadata-json", "metadata.json"])


def test_run_data_driven_requires_input_pkl():
    with pytest.raises(SystemExit):
        run_data_driven.main([])


def test_exact_legacy_context_qualifies_through_producer_query(tmp_path, monkeypatch):
    state_path, source_path, _state, input_sidecar = _legacy_context_inputs(tmp_path)
    _install_legacy_context_stubs(monkeypatch, input_sidecar)

    declaration, summary = run_data_driven._resolve_legacy_histogram_applicability(
        input_pkl=source_path,
        input_sidecar=input_sidecar,
        campaign_state_path=state_path,
        campaign_block_id="legacy_block",
    )

    assert declaration["analysis_mode"] == "all"
    assert set(declaration["families"]["lt"]["channels"].values()) == {
        "not_applicable"
    }
    assert summary["context_binding_method"] == "resolved_input_path"
    assert summary["source_pkl_sha256"] == "2" * 64
    assert summary["producer_topeft_commit"] == "1" * 40
    assert summary["producer_semantics_sha256"] == declaration[
        "producer_semantics_sha256"
    ]
    assert summary["producer_semantic_comparison"] == (
        "legacy_method_semantics_identical"
    )
    assert summary["historical_scope_source"] == "producer_commit_config"
    assert summary["analysis_mode"] == "all"
    assert summary["region"] == "SR"
    assert summary["category_groups"] == ["3l_onZ_tau"]
    assert summary["histogram_families"] == ["lt"]
    assert summary["years"] == ["2018"]
    assert summary["diagnostic_provenance_differences"] == []


@pytest.mark.parametrize(
    "mismatch,expected",
    [
        ("source", "context/source binding failed"),
        ("analysis_mode", "conflicting analysis modes"),
        ("artifact_family_scope", "requested family set disagrees"),
    ],
)
def test_legacy_context_mismatches_fail_closed(
    tmp_path,
    monkeypatch,
    mismatch,
    expected,
):
    state_path, source_path, state, input_sidecar = _legacy_context_inputs(tmp_path)
    families = ("njets",) if mismatch == "artifact_family_scope" else ("lt",)
    _install_legacy_context_stubs(monkeypatch, input_sidecar, families=families)
    block = state["blocks"][0]
    if mismatch == "source":
        different_source = tmp_path / "different_source.pkl.gz"
        block["expected_nominal_path"] = str(different_source)
        expected_sidecar = {
            **input_sidecar,
            "artifact": {
                **input_sidecar["artifact"],
                "pkl_sha256": "3" * 64,
            },
        }
        monkeypatch.setattr(
            run_data_driven,
            "read_histogram_sidecar",
            lambda _path: expected_sidecar,
        )
    elif mismatch == "analysis_mode":
        block["source_command_argv"].append("--tau-h-analysis")
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(RuntimeError, match=expected):
        run_data_driven._resolve_legacy_histogram_applicability(
            input_pkl=source_path,
            input_sidecar=input_sidecar,
            campaign_state_path=state_path,
            campaign_block_id="legacy_block",
        )


def test_legacy_context_same_path_does_not_require_redundant_source_sha(
    tmp_path,
    monkeypatch,
):
    state_path, source_path, _state, input_sidecar = _legacy_context_inputs(tmp_path)
    input_sidecar["artifact"].pop("pkl_sha256")
    _install_legacy_context_stubs(monkeypatch, input_sidecar)

    _declaration, summary = run_data_driven._resolve_legacy_histogram_applicability(
        input_pkl=source_path,
        input_sidecar=input_sidecar,
        campaign_state_path=state_path,
        campaign_block_id="legacy_block",
    )

    assert summary["context_binding_method"] == "resolved_input_path"
    assert summary["source_pkl_sha256"] is None


def test_legacy_context_accepts_sha_identified_scratch_copy(tmp_path, monkeypatch):
    state_path, source_path, state, input_sidecar = _legacy_context_inputs(tmp_path)
    scratch_path = tmp_path / "scratch_copy.pkl.gz"
    state["blocks"][0]["expected_nominal_path"] = str(source_path)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _install_legacy_context_stubs(monkeypatch, input_sidecar)

    _declaration, summary = run_data_driven._resolve_legacy_histogram_applicability(
        input_pkl=scratch_path,
        input_sidecar=input_sidecar,
        campaign_state_path=state_path,
        campaign_block_id="legacy_block",
    )

    assert summary["context_binding_method"] == "frozen_source_sha256"


def test_legacy_scope_uses_semantic_sets_and_historical_config(
    tmp_path,
    monkeypatch,
):
    state_path, source_path, state, input_sidecar = _legacy_context_inputs(tmp_path)
    command = state["blocks"][0]["source_command_argv"]
    command[:] = [
        "python",
        "run_analysis.py",
        "--all-analysis",
        "--hist-vars",
        "njets",
        "lt",
        "--sr",
        "--category-groups",
        "3l_onZ_tau",
        "4l",
        "-y",
        "2018",
    ]
    state["blocks"][0]["histograms"] = ["lt", "njets"]
    state["blocks"][0]["category_groups"] = ["4l", "3l_onZ_tau"]
    state["blocks"][0]["years"] = ["2018", "2017"]
    state["region"] = "CR"
    state["category_config_sha256"] = "0" * 64
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _install_legacy_context_stubs(
        monkeypatch,
        input_sidecar,
        families=("lt", "njets"),
    )
    monkeypatch.setattr(
        run_data_driven.analysis_processor,
        "load_category_config",
        lambda: (_ for _ in ()).throw(
            AssertionError("current category config must not be consulted")
        ),
    )

    declaration, summary = run_data_driven._resolve_legacy_histogram_applicability(
        input_pkl=source_path,
        input_sidecar=input_sidecar,
        campaign_state_path=state_path,
        campaign_block_id="legacy_block",
    )

    assert list(declaration["families"]) == ["lt", "njets"]
    assert summary["historical_scope_source"] == "producer_commit_config"
    assert summary["diagnostic_provenance_differences"] == [
        "region",
        "years",
    ]


def test_legacy_scope_prefers_explicit_retained_execution_config(
    tmp_path,
    monkeypatch,
):
    state_path, source_path, state, input_sidecar = _legacy_context_inputs(tmp_path)
    retained_path = tmp_path / "retained_ch_lst.json"
    retained_path.write_text(
        json.dumps(_legacy_category_config(lepton_channel="3l_onZ_1b_fwd")),
        encoding="utf-8",
    )
    state["blocks"][0]["retained_category_config_path"] = str(retained_path)
    state_path.write_text(json.dumps(state), encoding="utf-8")
    _install_legacy_context_stubs(monkeypatch, input_sidecar)

    declaration, summary = run_data_driven._resolve_legacy_histogram_applicability(
        input_pkl=source_path,
        input_sidecar=input_sidecar,
        campaign_state_path=state_path,
        campaign_block_id="legacy_block",
    )

    assert set(declaration["families"]["lt"]["channels"].values()) == {
        "applicable"
    }
    assert summary["historical_scope_source"] == "retained_execution_config"
    assert summary["historical_scope_locator"] == str(retained_path)


def test_legacy_producer_source_difference_passes_when_binary_decisions_match(
    tmp_path,
    monkeypatch,
):
    state_path, source_path, _state, input_sidecar = _legacy_context_inputs(tmp_path)
    producer_source = Path(run_data_driven.analysis_processor.__file__).read_text(
        encoding="utf-8"
    ).replace(
        "        skip_hist = False\n",
        "        skip_hist = False\n        pass\n",
        1,
    )
    _install_legacy_context_stubs(
        monkeypatch,
        input_sidecar,
        producer_source=producer_source,
    )

    _declaration, summary = run_data_driven._resolve_legacy_histogram_applicability(
        input_pkl=source_path,
        input_sidecar=input_sidecar,
        campaign_state_path=state_path,
        campaign_block_id="legacy_block",
    )

    assert summary["producer_semantic_comparison"] == (
        "relevant_binary_decisions_equal"
    )


def test_legacy_producer_source_difference_fails_only_for_material_decision(
    tmp_path,
    monkeypatch,
):
    state_path, source_path, _state, input_sidecar = _legacy_context_inputs(tmp_path)
    producer_source = Path(run_data_driven.analysis_processor.__file__).read_text(
        encoding="utf-8"
    ).replace(
        '            if (("lt" in dense_axis_name) and ("fwd" not in lep_chan)):\n'
        "                skip_hist = True\n",
        '            if (("lt" in dense_axis_name) and ("fwd" in lep_chan)):\n'
        "                skip_hist = True\n",
        1,
    )
    _install_legacy_context_stubs(
        monkeypatch,
        input_sidecar,
        producer_source=producer_source,
    )

    with pytest.raises(RuntimeError, match="applicability decisions differ"):
        run_data_driven._resolve_legacy_histogram_applicability(
            input_pkl=source_path,
            input_sidecar=input_sidecar,
            campaign_state_path=state_path,
            campaign_block_id="legacy_block",
        )


def _legacy_resolution_for_diagnostic():
    return {
        "analysis_mode": "all",
        "region": "SR",
        "category_groups": ["3l_onZ_tau"],
        "histogram_families": ["lt"],
        "years": ["2018"],
        "split_by_lepton_flavor": False,
    }


def _legacy_declaration_for_config(category_config):
    selected = run_data_driven._selected_category_dicts(
        analysis_mode="all",
        region="SR",
        category_groups=["3l_onZ_tau"],
        category_config=category_config,
    )
    return run_data_driven.analysis_processor.AnalysisProcessor.build_histogram_applicability(
        analysis_mode="all",
        runtime_families=("lt",),
        selected_category_dicts=selected,
        is_run3_values=(False,),
    )


def test_successful_legacy_structure_does_not_probe_local_config(monkeypatch):
    historical = _legacy_declaration_for_config(_legacy_category_config())
    monkeypatch.setattr(
        run_data_driven,
        "validate_histogram_artifact",
        lambda *_args, **_kwargs: {"status": "valid"},
    )
    monkeypatch.setattr(
        run_data_driven,
        "_current_local_category_config_candidate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local config must not be probed on the successful path")
        ),
    )

    result = run_data_driven._validate_legacy_histogram_artifact(
        "source.pkl.gz",
        historical,
        _legacy_resolution_for_diagnostic(),
    )

    assert result == {"status": "valid"}


def test_local_modified_config_is_diagnostic_only_after_contradiction(
    tmp_path,
    monkeypatch,
):
    historical = _legacy_declaration_for_config(_legacy_category_config())
    local_path = tmp_path / "ch_lst.json"
    local_path.write_text(
        json.dumps(_legacy_category_config(lepton_channel="3l_onZ_1b_fwd")),
        encoding="utf-8",
    )
    calls = []

    def _validate(_path, *, histogram_applicability):
        calls.append(histogram_applicability)
        states = set(
            histogram_applicability["families"]["lt"]["channels"].values()
        )
        if states == {"not_applicable"}:
            raise ValueError(
                "Family 'lt' is structurally present for not-applicable channel(s): test"
            )
        return {"status": "valid"}

    monkeypatch.setattr(run_data_driven, "validate_histogram_artifact", _validate)
    monkeypatch.setattr(
        run_data_driven,
        "_current_local_category_config_candidate",
        lambda _repository_root: (" M topeft/channels/ch_lst.json", local_path),
    )

    with pytest.raises(RuntimeError, match="historical_config_execution_ambiguity"):
        run_data_driven._validate_legacy_histogram_artifact(
            "source.pkl.gz",
            historical,
            _legacy_resolution_for_diagnostic(),
        )

    assert len(calls) == 2
    assert set(calls[0]["families"]["lt"]["channels"].values()) == {
        "not_applicable"
    }
    assert set(calls[1]["families"]["lt"]["channels"].values()) == {
        "applicable"
    }


def test_legacy_context_arguments_are_all_or_none():
    with pytest.raises(SystemExit):
        run_data_driven.main(
            [
                "--input-pkl",
                "source.pkl.gz",
                "--legacy-campaign-state",
                "state.json",
            ]
        )


def test_run_data_driven_legacy_dict_mode(tmp_path, monkeypatch):
    input_path = tmp_path / "input.pkl.gz"
    _write_histograms(input_path, {"seed": FakeHist(["dataUL18"])})
    output_path = tmp_path / "output.pkl.gz"

    DummyProducer.output_hist = {"njets": FakeHist(["flipsUL18", "ttbarUL18"])}
    monkeypatch.setattr(run_data_driven, "DataDrivenProducer", DummyProducer)

    monkeypatch.setattr(
        run_data_driven.utils,
        "dump_dict_streaming",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("streaming writer should not be used in legacy mode")
        ),
    )

    def _fake_dump_to_pkl(path, payload):
        with gzip.open(path, "wb") as stream:
            cloudpickle.dump(payload, stream)

    monkeypatch.setattr(run_data_driven.utils, "dump_to_pkl", _fake_dump_to_pkl)

    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--legacy-dict-mode",
        ]
    )

    assert DummyProducer.calls == [(str(input_path), str(output_path), False)]
    assert DummyProducer.get_calls == 1
    assert DummyProducer.iter_calls == 0
    result = _load_pkl(output_path)
    assert list(result["njets"].axes["process"]) == ["flipsUL18", "ttbarUL18"]


def test_data_driven_preserves_sumw2_companions_for_prompt_subtraction():
    base_hist = _build_data_driven_input_hist()
    sumw2_hist = _build_data_driven_input_hist()

    for histo, data_weight, prompt_weight in (
        (base_hist, 5.0, 2.0),
        (sumw2_hist, 25.0, 4.0),
    ):
        histo.fill(
            process="dataUL18",
            channel="2lss",
            systematic="nominal",
            appl="isAR_2lSS",
            met=np.array([0.5], dtype=float),
            weight=np.array([data_weight], dtype=float),
        )
        histo.fill(
            process="TTTo2L2Nu_centralUL18",
            channel="2lss",
            systematic="nominal",
            appl="isAR_2lSS",
            met=np.array([0.5], dtype=float),
            weight=np.array([prompt_weight], dtype=float),
        )

    producer = run_data_driven.DataDrivenProducer(
        {"met": base_hist, "met_sumw2": sumw2_hist},
        "unused-output.pkl.gz",
        iterator_mode=True,
    )
    result = dict(producer.iter_data_driven_histograms())

    assert "met" in result
    assert "met_sumw2" in result
    assert list(result["met"].axes["systematic"]) == ["nominal"]
    assert list(result["met_sumw2"].axes["systematic"]) == ["nominal"]
    assert list(result["met"].axes["process"]) == ["nonpromptUL18"]
    assert list(result["met_sumw2"].axes["process"]) == ["nonpromptUL18"]
    assert _single_bin_total(result["met"], "nonpromptUL18") == pytest.approx(3.0)
    assert _single_bin_total(result["met_sumw2"], "nonpromptUL18") == pytest.approx(29.0)


def test_run_data_driven_heartbeat(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "input.pkl.gz"
    _write_histograms(input_path, {"seed": FakeHist(["dataUL17"])})
    output_path = tmp_path / "output.pkl.gz"

    DummyProducer.output_hist = {
        "njets": FakeHist(["flipsUL17"]),
        "ht": FakeHist(["flipsUL17"]),
    }
    monkeypatch.setattr(run_data_driven, "DataDrivenProducer", DummyProducer)

    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--heartbeat-seconds",
            "0",
        ]
    )

    captured = capsys.readouterr().out
    assert "[run_data_driven] Processed" in captured
    assert "Finalized 2 histograms" in captured


def test_run_data_driven_quiet(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "input.pkl.gz"
    _write_histograms(input_path, {"seed": FakeHist(["dataUL17"])})
    output_path = tmp_path / "output.pkl.gz"

    DummyProducer.output_hist = {"njets": FakeHist(["flipsUL17"])}
    monkeypatch.setattr(run_data_driven, "DataDrivenProducer", DummyProducer)

    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--heartbeat-seconds",
            "0",
            "--quiet",
        ]
    )

    captured = capsys.readouterr().out
    assert "[run_data_driven]" not in captured


def test_run_data_driven_help_exposes_simplified_dd_report_contract():
    parser = run_data_driven._build_argument_parser()
    help_text = parser.format_help()

    assert "--dd-report" in help_text
    assert "--dd-report-md" in help_text
    assert "--dd-report-verbose" not in help_text

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--dd-report-verbose"])
    assert excinfo.value.code == 2


def test_run_data_driven_dd_report_nonprompt_and_sr(tmp_path, capsys):
    histogram = _fill_data_driven_histogram(
        [
            {
                "process": "dataUL18",
                "channel": "3l",
                "appl": "isAR_3l",
                "weight": 5.0,
            },
            {
                "process": "TTTo2L2Nu_centralUL18",
                "channel": "3l",
                "appl": "isAR_3l",
                "weight": 2.0,
            },
            {
                "process": "TTTo2L2Nu_centralUL18",
                "channel": "3l",
                "appl": "isSR_3l",
                "weight": 1.0,
            },
        ]
    )

    _run_with_dd_report(tmp_path, {"met": histogram})

    captured = capsys.readouterr().out
    assert "[dd-report] hist=met channel=3l" in captured
    assert "sr region=isSR_3l retained_total=1" in captured
    assert (
        "nonprompt region=isAR_3l out=nonpromptUL18 data_used=5 prompt_sub_used=2 result=3"
        in captured
    )


def test_run_data_driven_dd_report_flips_and_absent_regions(tmp_path, capsys):
    histogram = _fill_data_driven_histogram(
        [
            {
                "process": "dataUL18",
                "channel": "2lss",
                "appl": "isAR_2lSS_OS",
                "weight": 4.0,
            },
        ]
    )

    _run_with_dd_report(tmp_path, {"met": histogram})

    captured = capsys.readouterr().out
    assert "[dd-report] hist=met channel=2lss" in captured
    assert "sr region=isSR_2lSS absent" in captured
    assert "nonprompt region=isAR_2lSS absent" in captured
    assert "flips region=isAR_2lSS_OS out=flipsUL18 data_used=4 result=4" in captured


def test_run_data_driven_dd_report_missing_flips_region(tmp_path, capsys):
    histogram = _fill_data_driven_histogram(
        [
            {
                "process": "dataUL18",
                "channel": "2lss",
                "appl": "isAR_2lSS",
                "weight": 5.0,
            },
            {
                "process": "TTTo2L2Nu_centralUL18",
                "channel": "2lss",
                "appl": "isAR_2lSS",
                "weight": 2.0,
            },
        ]
    )

    _run_with_dd_report(tmp_path, {"met": histogram})

    captured = capsys.readouterr().out
    assert "nonprompt region=isAR_2lSS out=nonpromptUL18 data_used=5 prompt_sub_used=2 result=3" in captured
    assert "flips region=isAR_2lSS_OS absent" in captured


def test_run_data_driven_dd_report_empty_histogram(tmp_path, capsys):
    _run_with_dd_report(tmp_path, {"met": _build_data_driven_input_hist()})

    captured = capsys.readouterr().out
    assert "[dd-report] hist=met status=empty" in captured


def test_run_data_driven_dd_report_markdown_only_writes_detailed_file(tmp_path, capsys):
    histogram = _fill_data_driven_histogram(
        [
            {
                "process": "dataUL18",
                "channel": "3l",
                "appl": "isAR_3l",
                "weight": 5.0,
            },
            {
                "process": "TTTo2L2Nu_centralUL18",
                "channel": "3l",
                "appl": "isAR_3l",
                "weight": 2.0,
            },
            {
                "process": "TTTo2L2Nu_centralUL18",
                "channel": "3l",
                "appl": "isAR_3l",
                "systematic": "FFUp",
                "weight": 2.5,
            },
            {
                "process": "TTTo2L2Nu_centralUL18",
                "channel": "3l",
                "appl": "isAR_3l",
                "systematic": "JESUp",
                "weight": 9.0,
            },
        ]
    )
    report_path = tmp_path / "reports" / "dd_report.md"

    _run_histograms(
        tmp_path,
        {"met": histogram},
        "--dd-report-md",
        str(report_path),
    )

    captured = capsys.readouterr().out
    assert "[dd-report]" not in captured
    assert report_path.is_file()

    markdown = _read_text(report_path)
    assert "# Data-driven report" in markdown
    assert "## Histogram: `met`" in markdown
    assert "### Channel: `3l`" in markdown
    assert "- nonprompt region `isAR_3l` output `nonpromptUL18`" in markdown
    assert "  - data used: `5`" in markdown
    assert "  - prompt subtraction used: `2`" in markdown
    assert "  - result: `3`" in markdown
    assert "  - data sources: `dataUL18=5`" in markdown
    assert "  - prompt subtraction sources: `TTTo2L2Nu_centralUL18=2`" in markdown
    assert (
        "  - prompt subtraction systematics: `kept=FFUp,nominal`; `dropped=JESUp`"
        in markdown
    )


def test_run_data_driven_dd_report_stdout_is_compact_when_markdown_is_also_requested(
    tmp_path, capsys
):
    histogram = _fill_data_driven_histogram(
        [
            {
                "process": "dataUL18",
                "channel": "2lss",
                "appl": "isAR_2lSS",
                "weight": 5.0,
            },
            {
                "process": "TTTo2L2Nu_centralUL18",
                "channel": "2lss",
                "appl": "isAR_2lSS",
                "weight": 2.0,
            },
            {
                "process": "TTTo2L2Nu_centralUL18",
                "channel": "2lss",
                "appl": "isAR_2lSS",
                "systematic": "FFUp",
                "weight": 2.5,
            },
            {
                "process": "TTTo2L2Nu_centralUL18",
                "channel": "2lss",
                "appl": "isAR_2lSS",
                "systematic": "JESUp",
                "weight": 9.0,
            },
            {
                "process": "dataUL18",
                "channel": "2lss",
                "appl": "isAR_2lSS_OS",
                "weight": 4.0,
            },
        ]
    )
    report_path = tmp_path / "dd_report_detailed.md"

    output_path = _run_histograms(
        tmp_path,
        {"met": histogram},
        "--dd-report",
        "--dd-report-md",
        str(report_path),
        "--only-flips",
    )

    captured = capsys.readouterr().out
    assert "[dd-report] hist=met channel=2lss" in captured
    assert (
        "nonprompt region=isAR_2lSS out=nonpromptUL18 data_used=5 prompt_sub_used=2 result=3"
        in captured
    )
    assert "prompt_sub_sources" not in captured
    assert "data_sources:" not in captured
    markdown = _read_text(report_path)
    assert "- nonprompt region `isAR_2lSS` output `nonpromptUL18`" in markdown
    assert "  - prompt subtraction sources: `TTTo2L2Nu_centralUL18=2`" in markdown
    assert (
        "  - prompt subtraction systematics: `kept=FFUp,nominal`; `dropped=JESUp`"
        in markdown
    )
    assert "- flips region `isAR_2lSS_OS` output `flipsUL18`" in markdown

    result = _load_pkl(output_path)
    assert list(result["met"].axes["process"]) == ["flipsUL18"]


def test_run_data_driven_dd_report_zero_used_total(tmp_path, capsys):
    histogram = _fill_data_driven_histogram(
        [
            {
                "process": "dataUL18",
                "channel": "3l",
                "appl": "isAR_3l",
                "weight": 2.0,
            },
            {
                "process": "TTTo2L2Nu_centralUL18",
                "channel": "3l",
                "appl": "isAR_3l",
                "weight": 2.0,
            },
        ]
    )

    _run_with_dd_report(tmp_path, {"met": histogram})

    captured = capsys.readouterr().out
    assert "sr region=isSR_3l absent" in captured
    assert (
        "nonprompt region=isAR_3l out=nonpromptUL18 data_used=2 prompt_sub_used=2 result=0 zero_used_total"
        in captured
    )


def test_run_data_driven_dd_report_is_emitted_before_only_flips(tmp_path, capsys):
    histogram = _fill_data_driven_histogram(
        [
            {
                "process": "dataUL18",
                "channel": "2lss",
                "appl": "isAR_2lSS",
                "weight": 5.0,
            },
            {
                "process": "TTTo2L2Nu_centralUL18",
                "channel": "2lss",
                "appl": "isAR_2lSS",
                "weight": 2.0,
            },
            {
                "process": "dataUL18",
                "channel": "2lss",
                "appl": "isAR_2lSS_OS",
                "weight": 4.0,
            },
        ]
    )

    output_path = _run_with_dd_report(tmp_path, {"met": histogram}, "--only-flips")

    captured = capsys.readouterr().out
    assert "nonprompt region=isAR_2lSS out=nonpromptUL18 data_used=5 prompt_sub_used=2 result=3" in captured
    assert "flips region=isAR_2lSS_OS out=flipsUL18 data_used=4 result=4" in captured

    result = _load_pkl(output_path)
    assert list(result["met"].axes["process"]) == ["flipsUL18"]


def test_run_data_driven_dd_report_with_deprecated_envelope_fails_before_output(tmp_path, capsys):
    histogram = _fill_data_driven_histogram(
        [
            {
                "process": "dataUL18",
                "channel": "3l",
                "appl": "isAR_3l",
                "weight": 5.0,
            },
            {
                "process": "TTTo2L2Nu_centralUL18",
                "channel": "3l",
                "appl": "isAR_3l",
                "weight": 2.0,
            },
        ]
    )

    with pytest.raises(RuntimeError, match="combined renorm/fact envelope"):
        _run_with_dd_report(
            tmp_path,
            {"met": histogram},
            "--legacy-dict-mode",
            "--apply-renormfact-envelope",
            "--mem-report",
        )
    assert "[dd-report]" not in capsys.readouterr().out


def test_run_data_driven_dd_report_markdown_works_with_pkl_paths(tmp_path):
    histogram = _fill_data_driven_histogram(
        [
            {
                "process": "dataUL18",
                "channel": "3l",
                "appl": "isAR_3l",
                "weight": 5.0,
            },
            {
                "process": "TTTo2L2Nu_centralUL18",
                "channel": "3l",
                "appl": "isAR_3l",
                "weight": 2.0,
            },
        ]
    )
    input_path = tmp_path / "input.pkl.gz"
    output_path = tmp_path / "output.pkl.gz"
    report_path = tmp_path / "reports" / "metadata_dd_report.md"
    _write_histograms(input_path, {"met": histogram})

    run_data_driven.main(
        [
            "--input-pkl",
            str(input_path),
            "--output-pkl",
            str(output_path),
            "--dd-report-md",
            str(report_path),
            "--quiet",
        ]
    )

    markdown = _read_text(report_path)
    assert "## Histogram: `met`" in markdown
    assert "- nonprompt region `isAR_3l` output `nonpromptUL18`" in markdown
