import runpy
import sys
import types
from pathlib import Path
from unittest import mock

import coffea.processor as processor
import pytest

from analysis.topeft_run2 import analysis_processor as analysis_processor_module


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "analysis" / "topeft_run2" / "run_analysis.py"
SAMPLE_JSON = (
    REPO_ROOT
    / "input_samples"
    / "sample_jsons"
    / "test_samples"
    / "UL17_private_ttH_for_CI.json"
)


class processor_construction_boundary(RuntimeError):
    pass


def _mock_import_dependencies(monkeypatch):
    fake_data_driven = types.ModuleType("topeft.modules.dataDrivenEstimation")

    class dummy_producer:
        def __init__(self, *_, **__):
            pass

    fake_data_driven.DataDrivenProducer = dummy_producer
    monkeypatch.setitem(
        sys.modules,
        "topeft.modules.dataDrivenEstimation",
        fake_data_driven,
    )

    fake_hist_utils = types.ModuleType("topcoffea.modules.hist_utils")
    fake_hist_utils.iterate_hist_from_pkl = lambda *_, **__: iter(())
    monkeypatch.setitem(sys.modules, "topcoffea.modules.hist_utils", fake_hist_utils)

    fake_utils = types.ModuleType("topcoffea.modules.utils")
    fake_utils.get_hist_from_pkl = lambda *_, **__: {}
    fake_utils.dump_to_pkl = lambda *_, **__: None
    fake_utils.canonicalize_process_name = lambda name: name
    monkeypatch.setitem(sys.modules, "topcoffea.modules.utils", fake_utils)


def _run_cli_to_preflight_or_processor(
    monkeypatch,
    tmp_path,
    extra_args,
    *,
    use_real_sample,
    state=None,
):
    _mock_import_dependencies(monkeypatch)
    captured = {"processor_construction": 0, "executor_setup": 0}
    if state is not None:
        state["captured"] = captured

    class sentinel_analysis_processor:
        def __init__(self, *_, **__):
            captured["processor_construction"] += 1
            raise processor_construction_boundary()

    class forbidden_work_queue_executor:
        def __init__(self, *_, **__):
            captured["executor_setup"] += 1
            raise AssertionError("Work Queue setup must not precede the preflight")

    monkeypatch.setattr(
        analysis_processor_module,
        "AnalysisProcessor",
        sentinel_analysis_processor,
    )
    monkeypatch.setitem(
        sys.modules,
        "analysis_processor",
        analysis_processor_module,
    )
    monkeypatch.setattr(
        processor,
        "WorkQueueExecutor",
        forbidden_work_queue_executor,
        raising=False,
    )

    output_dir = tmp_path / "must-not-be-created"
    if state is not None:
        state["output_dir"] = output_dir
    input_path = SAMPLE_JSON if use_real_sample else tmp_path / "must-not-be-read.json"
    argv = [
        "run_analysis.py",
        str(input_path),
        "-p",
        str(output_dir),
        "-o",
        "preflight-test",
        "--skip-topcoffea-data-check",
        *extra_args,
    ]

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(SCRIPT_PATH.parent))
    try:
        with mock.patch.object(sys, "argv", argv):
            runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    return captured, output_dir


def test_explicit_incompatible_hist_fails_before_processor_executor_and_output(
    monkeypatch,
    tmp_path,
):
    state = {}
    with pytest.raises(SystemExit) as excinfo:
        _run_cli_to_preflight_or_processor(
            monkeypatch,
            tmp_path,
            [
                "--all-analysis",
                "--skip-sr",
                "--category-groups",
                "2l_CR",
                "--hist-list",
                "ptz_wtau",
                "--executor",
                "work_queue",
            ],
            use_real_sample=False,
            state=state,
        )

    message = str(excinfo.value)
    assert "ptz_wtau" in message
    assert "2l_CR" in message
    assert "ptz_wtau_channel_fill" in message
    assert "histogram_selection=explicit" in message
    assert "Processing was not started" in message
    assert state["captured"]["processor_construction"] == 0
    assert state["captured"]["executor_setup"] == 0
    assert not state["output_dir"].exists()


def test_implicit_default_with_nonprompt_fails_before_input_processing(
    monkeypatch,
    tmp_path,
):
    state = {}
    with pytest.raises(SystemExit) as excinfo:
        _run_cli_to_preflight_or_processor(
            monkeypatch,
            tmp_path,
            [
                "--all-analysis",
                "--skip-sr",
                "--category-groups",
                "2l_CR",
                "--do-np",
                "--executor",
                "work_queue",
            ],
            use_real_sample=False,
            state=state,
        )

    message = str(excinfo.value)
    assert "histogram_selection=implicit/default" in message
    assert "requested_data_driven_products=[nonprompt, flips]" in message
    assert "product_required_empty_family=yes" in message
    assert state["captured"]["processor_construction"] == 0
    assert state["captured"]["executor_setup"] == 0
    assert not state["output_dir"].exists()


def test_valid_explicit_request_reaches_processor_construction_boundary(
    monkeypatch,
    tmp_path,
):
    with pytest.raises(processor_construction_boundary):
        _run_cli_to_preflight_or_processor(
            monkeypatch,
            tmp_path,
            [
                "--all-analysis",
                "--skip-sr",
                "--category-groups",
                "2l_CR",
                "--hist-list",
                "met",
                "--executor",
                "futures",
            ],
            use_real_sample=True,
        )


def test_implicit_default_without_product_reaches_processor_boundary(
    monkeypatch,
    tmp_path,
):
    with pytest.raises(processor_construction_boundary):
        _run_cli_to_preflight_or_processor(
            monkeypatch,
            tmp_path,
            [
                "--all-analysis",
                "--skip-sr",
                "--category-groups",
                "2l_CR",
                "--executor",
                "futures",
            ],
            use_real_sample=True,
        )


def test_mixed_category_request_reaches_processor_construction_boundary(
    monkeypatch,
    tmp_path,
):
    with pytest.raises(processor_construction_boundary):
        _run_cli_to_preflight_or_processor(
            monkeypatch,
            tmp_path,
            [
                "--all-analysis",
                "--skip-sr",
                "--category-groups",
                "2l_CR",
                "1l_1tau_CRDY",
                "--hist-list",
                "ptz_wtau",
                "--executor",
                "futures",
            ],
            use_real_sample=True,
        )


def test_yaml_values_replace_overlapping_cli_values(monkeypatch, tmp_path):
    options_path = tmp_path / "recognized_options.yml"
    options_path.write_text(
        "hist_list:\n  - met\nuse_remote_env: false\n",
        encoding="utf-8",
    )

    with pytest.raises(processor_construction_boundary):
        _run_cli_to_preflight_or_processor(
            monkeypatch,
            tmp_path,
            [
                "--all-analysis",
                "--skip-sr",
                "--category-groups",
                "2l_CR",
                "--hist-list",
                "ptz_wtau",
                "--options",
                str(options_path),
                "--executor",
                "futures",
            ],
            use_real_sample=True,
        )


def test_unknown_yaml_keys_fail_before_processor_executor_and_output(
    monkeypatch,
    tmp_path,
):
    options_path = tmp_path / "unknown_options.yml"
    options_path.write_text(
        "zeta_typo: true\nalpha_typo: false\nuse_remote_env: false\n",
        encoding="utf-8",
    )
    state = {}

    with pytest.raises(
        ValueError,
        match=r"Unsupported YAML option key\(s\): alpha_typo, zeta_typo",
    ):
        _run_cli_to_preflight_or_processor(
            monkeypatch,
            tmp_path,
            ["--options", str(options_path), "--executor", "work_queue"],
            use_real_sample=False,
            state=state,
        )

    assert state["captured"]["processor_construction"] == 0
    assert state["captured"]["executor_setup"] == 0
    assert not state["output_dir"].exists()
