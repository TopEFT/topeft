import gzip
import importlib
import json
import runpy
import sys
import types
from pathlib import Path
from unittest import mock

import cloudpickle
import coffea.processor as processor
import pytest

from analysis.topeft_run2.analysis_processor import ANALYSIS_MODE_EXCLUSIVE_ERROR

_SAMPLE_JSON = Path("input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json")
_SCRIPT_PATH = Path("analysis/topeft_run2/run_analysis.py")
_EXPECTED_CR_BASE_HISTS = {
    "met",
    "l0conept",
    "l0eta",
    "njets",
}


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


def _run_run_analysis(monkeypatch, tmp_path, extra_cli_args, outname):
    output_dir = tmp_path / f"hist-output-{outname}"
    output_dir.mkdir()

    _mock_data_driven(monkeypatch)
    _mock_hist_utils(monkeypatch)
    _mock_topcoffea_utils(monkeypatch)

    def dummy_futures_executor(*, workers):
        return object()

    class DummyRunner:
        def __init__(self, exec_instance, *, schema=None, chunksize=None, maxchunks=None, **kwargs):
            self.exec_instance = exec_instance

        def __call__(self, fileset, treename, processor_instance):
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
        *extra_cli_args,
    ]

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        with mock.patch.object(sys, "argv", argv):
            runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    output_file = output_dir / f"{outname}.pkl.gz"
    with gzip.open(output_file, "rb") as fin:
        return cloudpickle.load(fin)


def test_hist_list_cr_includes_sumw2(monkeypatch, tmp_path):
    output = _run_run_analysis(monkeypatch, tmp_path, ["--hist-list", "cr"], "with-sumw2")

    expected_output_keys = set()
    for hist_name in _EXPECTED_CR_BASE_HISTS:
        expected_output_keys.add(hist_name)
        expected_output_keys.add(f"{hist_name}_sumw2")
        assert hist_name in output
        assert f"{hist_name}_sumw2" in output

    assert set(output) == expected_output_keys


def test_hist_list_cr_respects_no_sumw2(monkeypatch, tmp_path):
    output = _run_run_analysis(
        monkeypatch,
        tmp_path,
        ["--hist-list", "cr", "--no-sumw2"],
        "without-sumw2",
    )

    for hist_name in _EXPECTED_CR_BASE_HISTS:
        assert hist_name in output
        assert f"{hist_name}_sumw2" not in output

    assert set(output) == _EXPECTED_CR_BASE_HISTS


def test_custom_hist_list_accepts_fwd0eta(monkeypatch, tmp_path):
    output = _run_run_analysis(
        monkeypatch,
        tmp_path,
        ["--hist-list", "fwd0eta"],
        "custom-fwd0eta",
    )

    assert set(output) == {"fwd0eta", "fwd0eta_sumw2"}


def test_custom_hist_list_accepts_fwd0pt(monkeypatch, tmp_path):
    output = _run_run_analysis(
        monkeypatch,
        tmp_path,
        ["--hist-list", "fwd0pt"],
        "custom-fwd0pt",
    )

    assert set(output) == {"fwd0pt", "fwd0pt_sumw2"}


def test_np_postprocess_defer_creates_metadata(tmp_path):
    output_dir = tmp_path / "np-defer"
    output_dir.mkdir()
    outname = "np-defer"

    argv = [
        "run_analysis.py",
        str(_SAMPLE_JSON),
        "-x",
        "futures",
        "-o",
        outname,
        "-p",
        str(output_dir),
        "--pretend",
        "--do-np",
        "--np-postprocess=defer",
    ]

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit):
                runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    metadata_file = output_dir / f"{outname}_np.pkl.gz.metadata.json"
    assert metadata_file.is_file()
    np_pickle = output_dir / f"{outname}_np.pkl.gz"
    assert not np_pickle.exists()

    with open(metadata_file) as fin:
        payload = json.load(fin)

    assert payload["metadata_version"] == 2
    assert payload["np_postprocess"] == "defer"
    assert payload["pretend_mode"] is True
    assert payload["apply_renormfact_envelope"] is False
    assert payload["output_histogram"] == str(np_pickle)
    assert "run_data_driven.py --metadata-json" in payload["followup_command"]


def test_np_postprocess_defer_records_envelope_contract(tmp_path):
    output_dir = tmp_path / "np-defer-envelope"
    output_dir.mkdir()
    outname = "np-defer-envelope"

    argv = [
        "run_analysis.py",
        str(_SAMPLE_JSON),
        "-x",
        "futures",
        "-o",
        outname,
        "-p",
        str(output_dir),
        "--pretend",
        "--do-np",
        "--np-postprocess=defer",
        "--do-systs",
        "--do-renormfact-envelope",
    ]

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit):
                runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    metadata_file = output_dir / f"{outname}_np.pkl.gz.metadata.json"
    with open(metadata_file) as fin:
        payload = json.load(fin)

    assert payload["apply_renormfact_envelope"] is True


def test_missing_topcoffea_data_reports_guidance(monkeypatch):
    from topcoffea.modules.paths import topcoffea_path as real_topcoffea_path

    def fake_topcoffea_path(relpath):
        if relpath == "data/pileup/pileup_2016GH.root":
            raise FileNotFoundError("missing data bundle")
        return real_topcoffea_path(relpath)

    monkeypatch.setattr("topcoffea.modules.paths.topcoffea_path", fake_topcoffea_path)

    _mock_data_driven(monkeypatch)
    _mock_hist_utils(monkeypatch)
    _mock_topcoffea_utils(monkeypatch)

    argv = [
        "run_analysis.py",
        str(_SAMPLE_JSON),
        "-x",
        "futures",
        "--pretend",
    ]

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    message = str(excinfo.value)
    assert "scripts/install_topcoffea.sh" in message
    assert "--skip-topcoffea-data-check" in message


def test_empty_fileset_reports_clear_message(monkeypatch, tmp_path):
    empty_json = tmp_path / "empty.json"
    with open(_SAMPLE_JSON) as fin:
        payload = json.load(fin)

    payload["files"] = []
    with open(empty_json, "w") as fout:
        json.dump(payload, fout)

    _mock_data_driven(monkeypatch)
    _mock_hist_utils(monkeypatch)
    _mock_topcoffea_utils(monkeypatch)

    def dummy_futures_executor(*, workers):
        return object()

    class DummyRunner:
        def __init__(self, *_, **__):
            pass

        def __call__(self, *_, **__):
            raise AssertionError("Runner should not be invoked when there are no files")

    monkeypatch.setattr(processor, "futures_executor", dummy_futures_executor, raising=False)
    monkeypatch.setattr(processor, "Runner", DummyRunner)

    argv = ["run_analysis.py", str(empty_json), "-x", "futures"]

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    assert "No input files were available to process" in str(excinfo.value)


def test_worker_exception_is_reported(monkeypatch, tmp_path):
    _mock_data_driven(monkeypatch)

    def dummy_futures_executor(*, workers):
        return object()

    class DummyRunner:
        def __init__(self, exec_instance, *, schema=None, chunksize=None, maxchunks=None, **kwargs):
            self.exec_instance = exec_instance

        def __call__(self, fileset, treename, processor_instance):
            return {"out": {}, "exception": ValueError("forced worker crash")}

    monkeypatch.setattr(processor, "futures_executor", dummy_futures_executor, raising=False)
    monkeypatch.setattr(processor, "Runner", DummyRunner)

    argv = [
        "run_analysis.py",
        str(_SAMPLE_JSON),
        "-x",
        "futures",
        "--skip-topcoffea-data-check",
        "--hist-list",
        "cr",
    ]

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(RuntimeError) as excinfo:
                runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    assert "worker raised an exception" in str(excinfo.value)


def test_conflicting_analysis_mode_flags_raise_clear_error(monkeypatch):
    validator_calls = []

    def fake_validate_analysis_mode_flags(offz_3l_split, tau_h_analysis, fwd_analysis, all_analysis):
        validator_calls.append((offz_3l_split, tau_h_analysis, fwd_analysis, all_analysis))
        raise ValueError(ANALYSIS_MODE_EXCLUSIVE_ERROR)

    argv = [
        "run_analysis.py",
        str(_SAMPLE_JSON),
        "--offZ-3l-split",
        "--tau-h-analysis",
    ]

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        ap_cli = importlib.import_module("analysis_processor")
        monkeypatch.setattr(
            ap_cli,
            "validate_analysis_mode_flags",
            fake_validate_analysis_mode_flags,
            raising=True,
        )
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    assert str(excinfo.value) == ANALYSIS_MODE_EXCLUSIVE_ERROR
    assert validator_calls == [(True, True, False, False)]
