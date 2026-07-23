import inspect

from analysis.topeft_run2 import analysis_processor
from topeft.modules.axes import info as axes_info


def _processor_source_without_whitespace():
    source = inspect.getsource(analysis_processor.AnalysisProcessor.process)
    return "".join(source.split())


def test_fwd0eta_is_sourced_from_leading_forward_jet():
    source = _processor_source_without_whitespace()

    assert "fwd0=fwdJets[ak.argmax(fwdJets.pt,axis=-1,keepdims=True)]" in source
    assert 'varnames["fwd0eta"]=ak.flatten(fwd0.eta)' in source


def test_fwd0pt_is_sourced_from_leading_forward_jet():
    source = _processor_source_without_whitespace()

    assert "fwd0=fwdJets[ak.argmax(fwdJets.pt,axis=-1,keepdims=True)]" in source
    assert 'varnames["fwd0pt"]=ak.flatten(fwd0.pt)' in source
    assert 'varnames["fwd0pt"]=ak.flatten(j0.pt)' not in source
    assert 'varnames["fwd0pt"]=ak.flatten(goodJets.pt)' not in source


def test_j0eta_remains_sourced_from_leading_central_jet():
    source = _processor_source_without_whitespace()

    assert "j0=goodJets[ak.argmax(goodJets.pt,axis=-1,keepdims=True)]" in source
    assert 'varnames["j0pt"]=ak.flatten(j0.pt)' in source
    assert 'varnames["j0eta"]=ak.flatten(j0.eta)' in source
    assert 'varnames["j0pt"]=ak.flatten(fwd0.pt)' not in source
    assert 'varnames["j0eta"]=ak.flatten(fwd0.eta)' not in source


def test_fwd0eta_axis_is_forward_jet_specific():
    assert axes_info["fwd0eta"]["regular"] == (50, -5, 5)
    assert "forward jet" in axes_info["fwd0eta"]["label"]
    assert axes_info["j0eta"]["regular"] == (15, -3, 3)


def test_fwd0pt_axis_is_forward_jet_specific():
    assert axes_info["fwd0pt"]["regular"] == (10, 0, 200)
    assert "forward jet" in axes_info["fwd0pt"]["label"]
    assert axes_info["j0pt"]["regular"] == (15, 0, 300)
