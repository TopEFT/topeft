import inspect

from analysis.btagMCeff import btagMCeff
from analysis.topeft_run2 import analysis_processor
from analysis.topeft_run2 import analysis_processor_diboson


def test_main_processor_run3_uses_nanov12_tight_jet_id_for_central_and_forward_jets():
    source = inspect.getsource(analysis_processor.AnalysisProcessor.process)

    assert 'if is_run3:\n                jet_id_mask = tc_os.run3_nanoV12_ak4puppi_jet_id(cleanedJets, year, working_point="tight")' in source
    assert 'abs(cleanedJets.eta) < get_te_param("eta_j_cut")) & jet_id_mask' in source
    assert 'cleanedJets["isFwd"] = te_os.is_forward_jet_eta_banded' in source
    assert 'baseline_pt_cut=get_te_param("fwd_jet_pt_cut")' in source
    assert 'apply_eta_band_pt=te_os.resolve_fwd_eta_band_pt_apply(is_run3, self.fwd_eta_band_pt_apply)' in source
    assert 'quality_mask=jet_id_mask' in source
    assert 'else:\n                jet_id_mask = True' in source
    assert 'cleanedJets["isGood"] = tc_os.is_tight_jet' in source
    assert 'getattr(cleanedJets, jetptname) > 70.' not in source
    assert 'jetPtCut=50.' not in source


def test_diboson_processor_run3_uses_nanov12_tight_jet_id_for_central_and_forward_jets():
    source = inspect.getsource(analysis_processor_diboson.AnalysisProcessor.process)

    assert 'if is_run3:\n                jet_id_mask = tc_os.run3_nanoV12_ak4puppi_jet_id(cleanedJets, year, working_point="tight")' in source
    assert 'abs(cleanedJets.eta) < get_te_param("eta_j_cut")) & jet_id_mask' in source
    assert 'cleanedJets["isFwd"] = te_os.is_forward_jet_eta_banded' in source
    assert 'baseline_pt_cut=get_te_param("fwd_jet_pt_cut")' in source
    assert 'apply_eta_band_pt=te_os.resolve_fwd_eta_band_pt_apply(is_run3, self.fwd_eta_band_pt_apply)' in source
    assert 'quality_mask=jet_id_mask' in source
    assert 'else:\n                jet_id_mask = True' in source
    assert 'cleanedJets["isGood"] = tc_os.is_tight_jet' in source
    assert "getattr(cleanedJets, jetptname) > 50." not in source
    assert "jetPtCut=50." not in source
    assert "jetPtCut=40." not in source


def test_btag_mceff_aligns_run3_to_nanov12_tight_jet_id_and_preserves_run2_id_cut():
    source = inspect.getsource(btagMCeff.AnalysisProcessor.process)

    assert 'if is_run3:\n            jet_id_mask = tc_os.run3_nanoV12_ak4puppi_jet_id(j, year, working_point="tight")' in source
    assert 'abs(j.eta) < get_te_param("eta_j_cut")) & jet_id_mask' in source
    assert 'else:\n            j["isGood"] = tc_os.is_tight_jet' in source
    assert 'id_cut=get_te_param("jet_id_cut")' in source
    assert "id_cut=0" not in source
