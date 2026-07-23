import gzip
import importlib.util
from pathlib import Path

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
            process="ttbarUL18",
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
    producer.promptSubtractionSamples = {"ttbar"}

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
