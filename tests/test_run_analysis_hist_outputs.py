import gzip
import runpy
import sys
from pathlib import Path
from unittest import mock

import cloudpickle
import coffea.processor as processor

_SAMPLE_JSON = Path("input_samples/sample_jsons/test_samples/UL17_private_ttH_for_CI.json")
_SCRIPT_PATH = Path("analysis/topeft_run2/run_analysis.py")
_EXPECTED_BASE_HISTS = {
    "lj0pt",
    "met",
    "l0conept",
    "l0eta",
    "l1conept",
    "j0pt",
    "j0eta",
    "njets",
    "invmass",
}


def _run_run_analysis(monkeypatch, tmp_path, extra_cli_args, outname):
    output_dir = tmp_path / f"hist-output-{outname}"
    output_dir.mkdir()

    def dummy_futures_executor(*, workers):
        return object()

    class DummyRunner:
        def __init__(self, exec_instance, *, schema=None, chunksize=None, maxchunks=None, **kwargs):
            self.exec_instance = exec_instance

        def __call__(self, fileset, treename, processor_instance):
            return processor_instance.accumulator

    monkeypatch.setattr(processor, "futures_executor", dummy_futures_executor)
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

    for hist_name in _EXPECTED_BASE_HISTS:
        assert hist_name in output
        assert f"{hist_name}_sumw2" in output


def test_hist_list_cr_respects_no_sumw2(monkeypatch, tmp_path):
    output = _run_run_analysis(
        monkeypatch,
        tmp_path,
        ["--hist-list", "cr", "--no-sumw2"],
        "without-sumw2",
    )

    for hist_name in _EXPECTED_BASE_HISTS:
        assert hist_name in output
        assert f"{hist_name}_sumw2" not in output
