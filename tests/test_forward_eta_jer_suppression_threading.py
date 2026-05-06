import inspect
import runpy
import sys
from pathlib import Path
from unittest import mock

import pytest

from analysis.btagMCeff import btagMCeff
from analysis.topeft_run2 import analysis_processor
from analysis.topeft_run2 import analysis_processor_diboson
from topeft.modules import corrections as cor


_RUN_ANALYSIS_PATH = Path("analysis/topeft_run2/run_analysis.py")


def test_apply_jet_corrections_threads_forward_eta_suppression_to_factory(monkeypatch):
    calls = []

    class DummyJECStack:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def dummy_factory(name_map, jec_stack, run, **kwargs):
        calls.append(
            {
                "name_map": name_map,
                "jec_stack": jec_stack,
                "run": run,
                "kwargs": kwargs,
            }
        )
        return object()

    monkeypatch.setattr(cor, "JECStack", DummyJECStack)
    monkeypatch.setattr(cor, "CorrectedJetsFactory", dummy_factory)

    cor.ApplyJetCorrections(
        "2022",
        corr_type="jets",
        isData=False,
        era=None,
        run=123,
    )
    cor.ApplyJetCorrections(
        "2022",
        corr_type="jets",
        isData=False,
        era=None,
        run=123,
        suppress_forward_eta_stochastic_jer=True,
    )

    assert calls[0]["kwargs"]["suppress_forward_eta_stochastic_jer"] is False
    assert calls[1]["kwargs"]["suppress_forward_eta_stochastic_jer"] is True


def test_main_processor_stores_forward_eta_suppression_option_default_false():
    processor = analysis_processor.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=[],
    )

    assert processor.suppress_forward_eta_stochastic_jer is False


def test_main_processor_can_store_forward_eta_suppression_option_true():
    processor = analysis_processor.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=[],
        suppress_forward_eta_stochastic_jer=True,
    )

    assert processor.suppress_forward_eta_stochastic_jer is True


def test_processors_pass_forward_eta_suppression_to_apply_jet_corrections():
    main_source = inspect.getsource(analysis_processor.AnalysisProcessor.process)
    diboson_source = inspect.getsource(analysis_processor_diboson.AnalysisProcessor.process)
    btag_source = inspect.getsource(btagMCeff.AnalysisProcessor.process)

    expected = (
        "suppress_forward_eta_stochastic_jer="
        "self.suppress_forward_eta_stochastic_jer"
    )
    assert expected in main_source
    assert expected in diboson_source
    assert expected in btag_source


def test_secondary_processors_default_forward_eta_suppression_false():
    diboson_processor = analysis_processor_diboson.AnalysisProcessor(
        samples={},
        wc_names_lst=[],
        hist_lst=[],
    )
    btag_processor = btagMCeff.AnalysisProcessor(samples={})

    assert diboson_processor.suppress_forward_eta_stochastic_jer is False
    assert btag_processor.suppress_forward_eta_stochastic_jer is False


def test_run_analysis_help_exposes_forward_eta_suppression_flag(capsys):
    argv = ["run_analysis.py", "--help"]

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(_RUN_ANALYSIS_PATH.parent))
    try:
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_path(str(_RUN_ANALYSIS_PATH), run_name="__main__")
    finally:
        sys.path = original_sys_path

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out

    assert "--suppress-forward-eta-stochastic-jer" in help_text
    assert "requires JME/JERC approval" in help_text


def test_run_analysis_threads_forward_eta_suppression_to_main_processor():
    source = Path(_RUN_ANALYSIS_PATH).read_text()

    assert "suppress_forward_eta_stochastic_jer = args.suppress_forward_eta_stochastic_jer" in source
    assert '"suppress_forward_eta_stochastic_jer"' in source
    assert (
        "suppress_forward_eta_stochastic_jer=suppress_forward_eta_stochastic_jer"
        in source
    )
