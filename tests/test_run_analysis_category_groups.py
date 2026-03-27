import runpy
import sys
import types
from pathlib import Path
from unittest import mock

import coffea.processor as processor
import pytest

from analysis.topeft_run2 import analysis_processor as ap


_SAMPLE_JSON = Path("input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json")
_SCRIPT_PATH = Path("analysis/topeft_run2/run_analysis.py")


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


def _run_run_analysis_cli(monkeypatch, tmp_path, extra_cli_args, *, outname):
    output_dir = tmp_path / f"hist-output-{outname}"
    output_dir.mkdir()

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
            runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    return captured.get("processor_instance")


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
