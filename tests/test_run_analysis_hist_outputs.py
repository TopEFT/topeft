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
import numpy as np
import pytest

from analysis.topeft_run2.analysis_processor import ANALYSIS_MODE_EXCLUSIVE_ERROR
from topeft.modules.data_driven_products import data_driven_product_error

_SAMPLE_JSON = Path("input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json")
_SCRIPT_PATH = Path("analysis/topeft_run2/run_analysis.py")
_EXPECTED_CR_BASE_HISTS = {
    "ptz",
    "met",
    "lt",
}


def _mock_data_driven(monkeypatch):
    from topeft.modules.dataDrivenEstimation import (
        DataDrivenProducer as RealDataDrivenProducer,
    )

    fake_data_driven = types.ModuleType("topeft.modules.dataDrivenEstimation")

    class DummyProducer:
        def __init__(self, input_path, *args, **kwargs):
            self.input_path = input_path
            self._producer = RealDataDrivenProducer(input_path, *args, **kwargs)

        def dumpToPickle(self):
            return self._producer.dumpToPickle()

        def getDataDrivenHistogram(self):
            return self._producer.getDataDrivenHistogram()

        def get_transformation_context(self, artifact_kind):
            return self._producer.get_transformation_context(artifact_kind)

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


def _write_data_driven_sample_jsons(tmp_path):
    with open(_SAMPLE_JSON) as stream:
        template = json.load(stream)
    data_payload = dict(template)
    data_payload.update(
        {
            "histAxisName": "dataUL17",
            "isData": True,
            "WCnames": [],
        }
    )
    prompt_payload = dict(template)
    prompt_payload.update(
        {
            "histAxisName": "TTTo2L2Nu_centralUL17",
            "isData": False,
            "WCnames": [],
        }
    )
    data_path = tmp_path / "data_sample.json"
    prompt_path = tmp_path / "prompt_sample.json"
    data_path.write_text(json.dumps(data_payload), encoding="utf-8")
    prompt_path.write_text(json.dumps(prompt_payload), encoding="utf-8")
    return [str(data_path), str(prompt_path)]


def _write_signal_variant_jsons(tmp_path, processes):
    with open(_SAMPLE_JSON) as stream:
        template = json.load(stream)
    paths = []
    for index, process in enumerate(processes):
        payload = dict(template)
        payload.update(
            {
                "histAxisName": process,
                "isData": False,
                "WCnames": ["ctW"] if "private" in process else [],
                "year": "2022",
            }
        )
        path = tmp_path / f"signal_{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(str(path))
    return paths


def _full_diagnostics_options(tmp_path):
    path = tmp_path / "full_diagnostics.yml"
    path.write_text("sumw2_storage:\n  mode: full_diagnostics\n", encoding="utf-8")
    return str(path)


def _run_run_analysis(monkeypatch, tmp_path, extra_cli_args, outname):
    output_dir = tmp_path / f"hist-output-{outname}"
    output_dir.mkdir()

    _mock_data_driven(monkeypatch)
    if "--do-np" not in extra_cli_args:
        _mock_hist_utils(monkeypatch)
        _mock_topcoffea_utils(monkeypatch)

    def dummy_futures_executor(*, workers):
        return object()

    class DummyRunner:
        def __init__(self, exec_instance, *, schema=None, chunksize=None, maxchunks=None, **kwargs):
            self.exec_instance = exec_instance

        def __call__(self, fileset, treename, processor_instance):
            output = processor_instance.accumulator
            if "--do-np" in extra_cli_args:
                for key, histogram in output.items():
                    axis_names = [axis.name for axis in histogram.axes]
                    dense_names = [
                        name
                        for name in axis_names
                        if name not in {"process", "channel", "systematic", "appl"}
                        and name != "quadratic_term"
                    ]
                    if not dense_names:
                        continue
                    dense_values = {
                        name: np.asarray([0.5]) for name in dense_names
                    }
                    for process_name, appl, nominal_weight in (
                        ("dataUL17", "isAR_3l", 10.0),
                        ("TTTo2L2Nu_centralUL17", "isAR_3l", 3.0),
                        ("dataUL17", "isAR_2lSS_OS", 4.0),
                    ):
                        weight = (
                            nominal_weight**2
                            if key.endswith("_sumw2")
                            else nominal_weight
                        )
                        histogram.fill(
                            process=process_name,
                            channel="3l",
                            systematic="nominal",
                            appl=appl,
                            **dense_values,
                            weight=np.asarray([weight]),
                        )
            return output

    monkeypatch.setattr(processor, "futures_executor", dummy_futures_executor, raising=False)
    monkeypatch.setattr(processor, "Runner", DummyRunner)

    sample_paths = (
        _write_data_driven_sample_jsons(tmp_path)
        if "--do-np" in extra_cli_args
        else [str(_SAMPLE_JSON)]
    )
    argv = [
        "run_analysis.py",
        ",".join(sample_paths),
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
    output = _run_run_analysis(
        monkeypatch,
        tmp_path,
        [
            "--hist-list",
            "cr",
            "--options",
            _full_diagnostics_options(tmp_path),
        ],
        "with-sumw2",
    )

    expected_output_keys = set()
    for hist_name in _EXPECTED_CR_BASE_HISTS:
        expected_output_keys.add(f"{hist_name}__eft_nominal")
        expected_output_keys.add(f"{hist_name}_sumw2")
        assert f"{hist_name}__eft_nominal" in output
        assert hist_name not in output
        assert f"{hist_name}_sumw2" in output

    assert set(output) == expected_output_keys
    sidecar_path = (
        tmp_path
        / "hist-output-with-sumw2"
        / "with-sumw2.pkl.gz.metadata.json"
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["metadata_schema_version"] == 2
    assert sidecar["artifact"]["artifact_kind"] == "processor_output"
    assert sidecar["artifact"]["pkl_basename"] == "with-sumw2.pkl.gz"
    assert list(sidecar["sumw2_content_manifest"]["families"]) == sidecar[
        "sumw2_storage_provenance"
    ]["runtime_histogram_families"]
    assert sidecar["sumw2_storage_provenance"]["resolved_mode"] == "full_diagnostics"
    assert sidecar["sumw2_storage_provenance"]["signal_sample_profile"] == "unrestricted"
    assert sidecar["production_sample_contract"]["compatibility_validated"] is True


def test_hist_list_cr_respects_no_sumw2(monkeypatch, tmp_path):
    output = _run_run_analysis(
        monkeypatch,
        tmp_path,
        ["--hist-list", "cr", "--no-sumw2"],
        "without-sumw2",
    )

    for hist_name in _EXPECTED_CR_BASE_HISTS:
        assert f"{hist_name}__eft_nominal" in output
        assert hist_name not in output
        assert f"{hist_name}_sumw2" not in output

    assert set(output) == {f"{name}__eft_nominal" for name in _EXPECTED_CR_BASE_HISTS}


def test_custom_hist_list_accepts_fwd0eta(monkeypatch, tmp_path):
    output = _run_run_analysis(
        monkeypatch,
        tmp_path,
        ["--hist-list", "fwd0eta"],
        "custom-fwd0eta",
    )

    assert set(output) == {"fwd0eta__eft_nominal"}


def test_custom_hist_list_accepts_fwd0pt(monkeypatch, tmp_path):
    output = _run_run_analysis(
        monkeypatch,
        tmp_path,
        ["--hist-list", "fwd0pt"],
        "custom-fwd0pt",
    )

    assert set(output) == {"fwd0pt__eft_nominal"}


def test_np_postprocess_inline_writes_transformed_artifact_sidecar(
    monkeypatch, tmp_path
):
    _run_run_analysis(
        monkeypatch,
        tmp_path,
        ["--hist-list", "cr", "--do-np", "--np-postprocess=inline"],
        "inline-np",
    )
    output_path = tmp_path / "hist-output-inline-np" / "inline-np_np.pkl.gz"
    sidecar_path = Path(f"{output_path}.metadata.json")
    assert output_path.is_file()
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["artifact"]["artifact_kind"] == "nonprompt_output"
    assert sidecar["lineage"]["inputs"][0]["pkl_basename"] == "inline-np.pkl.gz"
    assert sidecar["requested_data_driven_products"]["products"] == {
        "nonprompt": {"enabled": True},
        "flips": {"enabled": True},
    }
    assert sidecar["resolved_data_driven_contract"]["contract_version"] == 3
    assert sidecar["resolved_data_driven_contract"]["products"]["nonprompt"][
        "generated_outputs"
    ] == {
        "nonpromptUL17": {
            "year": "UL17",
            "source_contributors": {
                "data": ["dataUL17"],
                "prompt_mc": ["TTTo2L2Nu_centralUL17"],
            },
            "required_source_sumw2_processes": [
                "TTTo2L2Nu_centralUL17",
                "dataUL17",
            ],
        }
    }


def test_incomplete_requested_product_fails_before_processor_construction(
    monkeypatch, tmp_path
):
    sample_paths = _write_data_driven_sample_jsons(tmp_path)
    options_path = tmp_path / "incomplete.yml"
    options_path.write_text(
        """sumw2_storage:
  mode: full_custom
  rules:
    - process_names: [dataUL17]
      variables: [met]
data_driven_products:
  nonprompt:
    enabled: true
    source_contributors:
      data:
        process_names: [dataUL17]
      prompt_mc:
        process_names: [TTTo2L2Nu_centralUL17]
  flips:
    enabled: false
""",
        encoding="utf-8",
    )
    processor_calls = []

    def forbidden_processor(*args, **kwargs):
        processor_calls.append((args, kwargs))
        raise AssertionError("processor construction must not begin")

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        processor_module = importlib.import_module("analysis_processor")
        monkeypatch.setattr(
            processor_module,
            "AnalysisProcessor",
            forbidden_processor,
        )
        argv = [
            "run_analysis.py",
            ",".join(sample_paths),
            "--executor",
            "futures",
            "--hist-list",
            "met",
            "--options",
            str(options_path),
            "--skip-topcoffea-data-check",
        ]
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as error_info:
                runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
            assert "SUMW2-PROFILE-E005" in str(error_info.value)
            assert "affected_data_driven_product='nonprompt'" in str(
                error_info.value
            )
            assert "Recommended correction" in str(error_info.value)
    finally:
        sys.path = original_sys_path

    assert processor_calls == []


@pytest.mark.parametrize(
    "mode,signal_process",
    [
        ("production", "tllq_private2022"),
        ("production_central", "tZq_central2022"),
    ],
)
def test_required_signal_omission_fails_before_processor_construction(
    monkeypatch,
    tmp_path,
    mode,
    signal_process,
):
    sample_paths = _write_data_driven_sample_jsons(tmp_path)
    sample_paths.extend(_write_signal_variant_jsons(tmp_path, [signal_process]))
    options_path = tmp_path / f"missing_required_signal_{mode}.yml"
    options_path.write_text(
        f"""sumw2_storage:
  mode: {mode}
  rules:
    - process_names: [dataUL17, TTTo2L2Nu_centralUL17, {signal_process}]
      variables: [met]
data_driven_products:
  nonprompt:
    enabled: true
    source_contributors:
      data:
        process_names: [dataUL17]
      prompt_mc:
        process_names: [TTTo2L2Nu_centralUL17]
  flips:
    enabled: false
""",
        encoding="utf-8",
    )
    processor_calls = []

    def forbidden_processor(*args, **kwargs):
        processor_calls.append((args, kwargs))
        raise AssertionError("processor construction must not begin")

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        processor_module = importlib.import_module("analysis_processor")
        monkeypatch.setattr(
            processor_module,
            "AnalysisProcessor",
            forbidden_processor,
        )
        argv = [
            "run_analysis.py",
            ",".join(sample_paths),
            "--executor",
            "futures",
            "--hist-list",
            "met",
            "--options",
            str(options_path),
            "--skip-topcoffea-data-check",
            "--sample-universe-wrapper",
            "required-signal-test",
        ]
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as error_info:
                runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
        message = str(error_info.value)
        assert "SUMW2-PROFILE-E007" in message
        assert f"resolved_mode='{mode}'" in message
        assert signal_process in message
        assert "metadata_source='explicit'" in message
        assert "Recommended correction" in message
    finally:
        sys.path = original_sys_path

    assert processor_calls == []


@pytest.mark.parametrize(
    "mode,processes,error_id",
    [
        ("production", ["tZq_central2022"], "SUMW2-PROFILE-E001"),
        ("production_central", ["tllq_private2022"], "SUMW2-PROFILE-E002"),
        (
            "full_diagnostics",
            ["tllq_private2022", "tZq_central2022"],
            "SUMW2-PROFILE-E003",
        ),
    ],
)
def test_cfg_mode_profile_mismatch_fails_before_processor_construction(
    monkeypatch,
    tmp_path,
    mode,
    processes,
    error_id,
):
    sample_paths = _write_signal_variant_jsons(tmp_path, processes)
    options_path = tmp_path / f"{mode}.yml"
    if mode == "full_diagnostics":
        storage = "sumw2_storage:\n  mode: full_diagnostics\n"
    else:
        storage = (
            f"sumw2_storage:\n  mode: {mode}\n  rules:\n"
            f"    - process_names: [{processes[0]}]\n      variables: [met]\n"
        )
    options_path.write_text(storage, encoding="utf-8")
    processor_calls = []

    def forbidden_processor(*args, **kwargs):
        processor_calls.append((args, kwargs))
        raise AssertionError("processor construction must not begin")

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        processor_module = importlib.import_module("analysis_processor")
        monkeypatch.setattr(
            processor_module,
            "AnalysisProcessor",
            forbidden_processor,
        )
        argv = [
            "run_analysis.py",
            ",".join(sample_paths),
            "--executor",
            "futures",
            "--hist-list",
            "met",
            "--options",
            str(options_path),
            "--skip-topcoffea-data-check",
            "--sample-universe-wrapper",
            "synthetic-profile-test",
        ]
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as error_info:
                runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
            message = str(error_info.value)
            assert error_id in message
            assert "wrapper='synthetic-profile-test'" in message
            assert "Recommended correction" in message
    finally:
        sys.path = original_sys_path

    assert processor_calls == []


def test_orphan_prompt_year_fails_before_processor_construction(
    monkeypatch, tmp_path
):
    with open(_SAMPLE_JSON) as stream:
        template = json.load(stream)
    sample_paths = []
    for filename, process, is_data in (
        ("data_18.json", "dataUL18", True),
        ("prompt_17.json", "TTTo2L2Nu_centralUL17", False),
    ):
        payload = dict(template)
        payload.update(
            {
                "histAxisName": process,
                "isData": is_data,
                "WCnames": [],
            }
        )
        if process.endswith("UL18"):
            payload["year"] = "2018"
            payload["files"] = [
                filename.replace("UL17", "UL18")
                for filename in payload["files"]
            ]
        path = tmp_path / filename
        path.write_text(json.dumps(payload), encoding="utf-8")
        sample_paths.append(str(path))
    options_path = tmp_path / "orphan_year.yml"
    options_path.write_text(
        """sumw2_storage:
  mode: full_diagnostics
data_driven_products:
  nonprompt:
    enabled: true
    source_contributors:
      data:
        process_names: [dataUL18]
      prompt_mc:
        process_names: [TTTo2L2Nu_centralUL17]
  flips:
    enabled: false
""",
        encoding="utf-8",
    )
    processor_calls = []

    def forbidden_processor(*args, **kwargs):
        processor_calls.append((args, kwargs))
        raise AssertionError("processor construction must not begin")

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_SCRIPT_PATH.parent))
    try:
        processor_module = importlib.import_module("analysis_processor")
        monkeypatch.setattr(
            processor_module,
            "AnalysisProcessor",
            forbidden_processor,
        )
        argv = [
            "run_analysis.py",
            ",".join(sample_paths),
            "--executor",
            "futures",
            "--hist-list",
            "met",
            "--options",
            str(options_path),
            "--skip-topcoffea-data-check",
        ]
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(
                data_driven_product_error,
                match=r"orphan_years=\['UL17'\].*Recommended correction",
            ):
                runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    assert processor_calls == []


def test_np_postprocess_defer_prints_pkl_only_followup(tmp_path, capsys):
    output_dir = tmp_path / "np-defer"
    output_dir.mkdir()
    outname = "np-defer"

    sample_paths = _write_data_driven_sample_jsons(tmp_path)
    argv = [
        "run_analysis.py",
        ",".join(sample_paths),
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
    assert not metadata_file.exists()
    np_pickle = output_dir / f"{outname}_np.pkl.gz"
    assert not np_pickle.exists()
    output = capsys.readouterr().out
    assert "run_data_driven.py --input-pkl" in output
    assert "--output-pkl" in output
    assert "metadata-json" not in output


def test_np_postprocess_defer_rejects_deprecated_envelope_before_work(tmp_path, capsys):
    output_dir = tmp_path / "np-defer-envelope"
    output_dir.mkdir()
    outname = "np-defer-envelope"

    sample_paths = _write_data_driven_sample_jsons(tmp_path)
    argv = [
        "run_analysis.py",
        ",".join(sample_paths),
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
            with pytest.raises(RuntimeError, match="combined renorm/fact envelope"):
                runpy.run_path(str(_SCRIPT_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    assert "run_data_driven.py" not in capsys.readouterr().out
    assert not (output_dir / f"{outname}_np.pkl.gz.metadata.json").exists()
    assert not (output_dir / f"{outname}.pkl.gz").exists()


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
