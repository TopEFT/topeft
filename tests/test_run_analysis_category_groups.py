import hashlib
import json
import runpy
import subprocess
import sys
import types
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import coffea.processor as processor
import pytest

from analysis.topeft_run2 import analysis_processor as ap


_SAMPLE_JSON = Path("input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json")
_SCRIPT_PATH = Path("analysis/topeft_run2/run_analysis.py")
_REVIEWED_OBSERVABILITY_COMMIT = "8e2d3a77bab6d35365ab499b2566797729f57cd1"


def _mock_data_driven(monkeypatch):
    fake_data_driven = types.ModuleType("topeft.modules.dataDrivenEstimation")

    class DummyProducer:
        def __init__(self, *_, **__):
            pass

        def dumpToPickle(self):
            return None

    fake_data_driven.DataDrivenProducer = DummyProducer
    monkeypatch.setitem(sys.modules, "topeft.modules.dataDrivenEstimation", fake_data_driven)


def _mock_hist_utils(monkeypatch):
    fake_hist_utils = types.ModuleType("topcoffea.modules.hist_utils")

    def _dummy_iterate_hist_from_pkl(*args, **kwargs):
        return iter(())

    fake_hist_utils.iterate_hist_from_pkl = _dummy_iterate_hist_from_pkl
    monkeypatch.setitem(sys.modules, "topcoffea.modules.hist_utils", fake_hist_utils)


def _mock_topcoffea_utils(monkeypatch):
    fake_utils = types.ModuleType("topcoffea.modules.utils")

    def _dummy_get_hist_from_pkl(*args, **kwargs):
        return {}

    def _dummy_dump_to_pkl(*args, **kwargs):
        return None

    def _dummy_canonicalize_process_name(name):
        return name

    fake_utils.get_hist_from_pkl = _dummy_get_hist_from_pkl
    fake_utils.dump_to_pkl = _dummy_dump_to_pkl
    fake_utils.canonicalize_process_name = _dummy_canonicalize_process_name
    monkeypatch.setitem(sys.modules, "topcoffea.modules.utils", fake_utils)


def _run_run_analysis_cli(
    monkeypatch,
    tmp_path,
    extra_cli_args,
    *,
    outname,
    return_output_dir=False,
    runner_result_factory=None,
    script_path=_SCRIPT_PATH,
):
    output_dir = tmp_path / f"hist-output-{outname}"
    output_dir.mkdir(parents=True)

    _mock_data_driven(monkeypatch)
    _mock_hist_utils(monkeypatch)
    _mock_topcoffea_utils(monkeypatch)

    captured = {}

    def dummy_futures_executor(*, workers):
        return object()

    class DummyRunner:
        def __init__(self, exec_instance, *, schema=None, chunksize=None, maxchunks=None, **kwargs):
            self.exec_instance = exec_instance

        def __call__(self, fileset, treename, processor_instance):
            captured["processor_instance"] = processor_instance
            if runner_result_factory is not None:
                return runner_result_factory(processor_instance)
            return processor_instance.accumulator

    monkeypatch.setattr(processor, "futures_executor", dummy_futures_executor, raising=False)
    monkeypatch.setattr(processor, "Runner", DummyRunner)

    argv = [
        "run_analysis.py",
        str(_SAMPLE_JSON),
        "-x",
        "futures",
        "-o",
        outname,
        "-p",
        str(output_dir),
        "--skip-topcoffea-data-check",
        *extra_cli_args,
    ]

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        with mock.patch.object(sys, "argv", argv):
            runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.path = original_sys_path

    processor_instance = captured.get("processor_instance")
    if return_output_dir:
        return processor_instance, output_dir
    return processor_instance


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _historical_preinstrumentation_script(tmp_path):
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "git",
            "show",
            f"{_REVIEWED_OBSERVABILITY_COMMIT}^:analysis/topeft_run2/run_analysis.py",
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    historical_script = tmp_path / "run_analysis_preinstrumentation.py"
    historical_script.write_text(result.stdout, encoding="utf-8")
    return historical_script


def _histogram_container_semantics(pkl_path):
    import cloudpickle

    with gzip.open(pkl_path, "rb") as artifact_file:
        histograms = cloudpickle.load(artifact_file)
    return {
        key: {
            "axis_names": tuple(axis.name for axis in histogram.axes),
            "axis_definitions": tuple(repr(axis) for axis in histogram.axes),
            "serialized_content_sha256": hashlib.sha256(
                cloudpickle.dumps(histogram)
            ).hexdigest(),
        }
        for key, histogram in sorted(histograms.items())
    }


def test_post_runner_failure_persists_manager_diagnostic_and_reraises(
    monkeypatch, tmp_path
):
    with pytest.raises(RuntimeError, match="no results were returned"):
        _run_run_analysis_cli(
            monkeypatch,
            tmp_path,
            ["--skip-cr"],
            outname="manager-diagnostic-controlled-failure",
            runner_result_factory=lambda processor_instance: None,
        )

    output_dir = tmp_path / "hist-output-manager-diagnostic-controlled-failure"
    diagnostics = list(output_dir.glob("*.manager_failure.*.json"))
    assert len(diagnostics) == 1
    payload = json.loads(diagnostics[0].read_text(encoding="utf-8"))
    assert payload["phase"] == "runner_result_inspection"
    assert payload["exception"]["type"] == "RuntimeError"
    assert "no results were returned" in payload["exception"]["message"]
    assert "Processing failed because no results were returned" in payload["exception"]["traceback"]
    assert payload["run"]["executor"] == "futures"
    assert payload["manager_resources"]["pid"] > 0
    assert "max_rss_kib" in payload["manager_resources"]
    timestamp = datetime.fromisoformat(payload["timestamp_utc"])
    assert timestamp.utcoffset() == timedelta(0)


def test_nominal_publication_failure_persists_diagnostic_and_reraises(
    monkeypatch, tmp_path
):
    def controlled_artifact_writer(*args, **kwargs):
        raise OSError("controlled nominal publication failure")

    monkeypatch.setattr(
        "topeft.modules.histogram_artifact.write_histogram_artifact",
        controlled_artifact_writer,
    )

    with pytest.raises(OSError, match="controlled nominal publication failure"):
        _run_run_analysis_cli(
            monkeypatch,
            tmp_path,
            ["--skip-cr"],
            outname="manager-diagnostic-nominal-publication-failure",
        )

    output_dir = tmp_path / "hist-output-manager-diagnostic-nominal-publication-failure"
    diagnostics = list(output_dir.glob("*.manager_failure.*.json"))
    assert len(diagnostics) == 1
    payload = _read_json(diagnostics[0])
    assert payload["phase"] == "nominal_artifact_publication"
    assert payload["exception"]["type"] == "OSError"
    assert "controlled nominal publication failure" in payload["exception"]["message"]
    assert "controlled_artifact_writer" in payload["exception"]["traceback"]
    timestamp = datetime.fromisoformat(payload["timestamp_utc"])
    assert timestamp.utcoffset() == timedelta(0)
    assert payload["manager_resources"]["pid"] > 0
    assert "max_rss_kib" in payload["manager_resources"]
    assert not (output_dir / "manager-diagnostic-nominal-publication-failure.pkl.gz").exists()


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_deliberate_interruptions_do_not_write_manager_diagnostics(
    monkeypatch, tmp_path, interrupt_type
):
    def raise_deliberate_interruption(processor_instance):
        raise interrupt_type()

    with pytest.raises(interrupt_type):
        _run_run_analysis_cli(
            monkeypatch,
            tmp_path,
            ["--skip-cr"],
            outname=f"manager-diagnostic-{interrupt_type.__name__.lower()}",
            runner_result_factory=raise_deliberate_interruption,
        )

    assert not list(tmp_path.rglob("*.manager_failure.*.json"))


def test_successful_run_keeps_nominal_artifact_contract_without_diagnostic(
    monkeypatch, tmp_path
):
    outname = "manager-diagnostic-success"
    _, output_dir = _run_run_analysis_cli(
        monkeypatch,
        tmp_path,
        ["--skip-cr"],
        outname=outname,
        return_output_dir=True,
    )

    assert (output_dir / f"{outname}.pkl.gz").is_file()
    assert (output_dir / f"{outname}.pkl.gz.metadata.json").is_file()
    assert not list(output_dir.glob("*.manager_failure.*.json"))


def test_successful_run_matches_preinstrumentation_semantic_oracle(monkeypatch, tmp_path):
    outname = "manager-diagnostic-semantic-oracle"
    historical_script = _historical_preinstrumentation_script(tmp_path)
    _, historical_output_dir = _run_run_analysis_cli(
        monkeypatch,
        tmp_path / "historical",
        ["--skip-cr"],
        outname=outname,
        return_output_dir=True,
        script_path=historical_script,
    )
    _, current_output_dir = _run_run_analysis_cli(
        monkeypatch,
        tmp_path / "current",
        ["--skip-cr"],
        outname=outname,
        return_output_dir=True,
    )

    historical_pkl = historical_output_dir / f"{outname}.pkl.gz"
    current_pkl = current_output_dir / f"{outname}.pkl.gz"
    historical_sidecar = _read_json(historical_output_dir / f"{outname}.pkl.gz.metadata.json")
    current_sidecar = _read_json(current_output_dir / f"{outname}.pkl.gz.metadata.json")

    assert _histogram_container_semantics(current_pkl) == _histogram_container_semantics(
        historical_pkl
    )
    current_artifact = dict(current_sidecar["artifact"])
    historical_artifact = dict(historical_sidecar["artifact"])
    current_artifact.pop("pkl_sha256")
    historical_artifact.pop("pkl_sha256")
    assert current_artifact == historical_artifact
    assert current_sidecar["sumw2_storage_provenance"] == historical_sidecar[
        "sumw2_storage_provenance"
    ]
    assert current_sidecar["sumw2_content_manifest"] == historical_sidecar[
        "sumw2_content_manifest"
    ]
    assert not list(current_output_dir.glob("*.manager_failure.*.json"))


def test_category_groups_accepts_multiple_valid_groups_in_resolved_block(
    monkeypatch, tmp_path, capsys
):
    processor_instance = _run_run_analysis_cli(
        monkeypatch,
        tmp_path,
        ["--all-analysis", "--skip-cr", "--category-groups", "3l_fwd", "4l"],
        outname="category-groups-all-mode",
    )

    assert processor_instance.sr_category_dict_name == "ALL_CH_LST_SR"
    assert list(processor_instance.sr_category_dict.keys()) == ["3l_fwd", "4l"]
    assert processor_instance.cr_category_dict == {}

    stdout = capsys.readouterr().out
    assert "Resolved SR ch_lst.json block: ALL_CH_LST_SR" in stdout
    assert "Selected SR category groups: 3l_fwd, 4l" in stdout


def test_category_groups_unknown_name_fails_clearly(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        _run_run_analysis_cli(
            monkeypatch,
            tmp_path,
            ["--pretend", "--skip-cr", "--category-groups", "not_a_group"],
            outname="category-groups-unknown",
        )

    message = str(excinfo.value)
    assert "Unknown or incompatible category group(s): not_a_group" in message
    assert "TOP22_006_CH_LST_SR" in message


def test_category_groups_incompatible_name_fails_clearly(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        _run_run_analysis_cli(
            monkeypatch,
            tmp_path,
            ["--pretend", "--skip-cr", "--tau-h-analysis", "--category-groups", "2l"],
            outname="category-groups-incompatible",
        )

    message = str(excinfo.value)
    assert "Unknown or incompatible category group(s): 2l" in message
    assert "TAU_CH_LST_SR" in message


def test_category_groups_duplicates_are_normalized(monkeypatch, tmp_path, capsys):
    processor_instance = _run_run_analysis_cli(
        monkeypatch,
        tmp_path,
        ["--all-analysis", "--skip-cr", "--category-groups", "4l", "4l", "3l_fwd", "4l"],
        outname="category-groups-deduped",
    )

    assert list(processor_instance.sr_category_dict.keys()) == ["4l", "3l_fwd"]

    stdout = capsys.readouterr().out
    assert "Requested category groups (deduplicated user order): 4l, 3l_fwd" in stdout


def test_category_groups_no_option_uses_all_groups(monkeypatch, tmp_path, capsys):
    processor_instance = _run_run_analysis_cli(
        monkeypatch,
        tmp_path,
        ["--skip-cr"],
        outname="category-groups-default",
    )

    assert list(processor_instance.sr_category_dict.keys()) == ["2l", "3l", "4l"]

    stdout = capsys.readouterr().out
    assert "no --category-groups filter requested" in stdout
    assert "Selected SR category groups: all (2l, 3l, 4l)" in stdout


def test_category_groups_mixed_sr_cr_allows_sr_only_match(monkeypatch, tmp_path, capsys):
    sr_block_name, cr_block_name = ap.resolve_category_dict_names(False, False, False, False)
    category_config = ap.load_category_config()

    assert "4l" in category_config[sr_block_name]
    assert "4l" not in category_config[cr_block_name]

    processor_instance = _run_run_analysis_cli(
        monkeypatch,
        tmp_path,
        ["--category-groups", "4l"],
        outname="category-groups-mixed-sr-cr",
    )

    assert processor_instance.sr_category_dict_name == sr_block_name
    assert processor_instance.cr_category_dict_name == cr_block_name
    assert list(processor_instance.sr_category_dict.keys()) == ["4l"]
    assert processor_instance.cr_category_dict == {}

    stdout = capsys.readouterr().out
    assert "Requested category groups (deduplicated user order): 4l" in stdout
    assert f"Resolved SR ch_lst.json block: {sr_block_name}" in stdout
    assert f"Resolved CR ch_lst.json block: {cr_block_name}" in stdout
    assert "Selected SR category groups: 4l" in stdout
    assert "Selected CR category groups: <none>" in stdout


def test_category_groups_filter_downstream_active_blocks(monkeypatch, tmp_path, capsys):
    processor_instance = _run_run_analysis_cli(
        monkeypatch,
        tmp_path,
        ["--category-groups", "4l"],
        outname="category-groups-downstream",
    )

    assert processor_instance.sr_category_dict_name == "TOP22_006_CH_LST_SR"
    assert processor_instance.cr_category_dict_name == "CH_LST_CR"
    assert list(processor_instance.sr_category_dict.keys()) == ["4l"]
    assert processor_instance.cr_category_dict == {}

    stdout = capsys.readouterr().out
    assert "Resolved SR ch_lst.json block: TOP22_006_CH_LST_SR" in stdout
    assert "Resolved CR ch_lst.json block: CH_LST_CR" in stdout
    assert "Selected SR category groups: 4l" in stdout
    assert "Selected CR category groups: <none>" in stdout
