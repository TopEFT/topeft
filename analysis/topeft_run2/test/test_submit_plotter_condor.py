"""Public control and batch worker stay separate after launcher retirement."""
from pathlib import Path
import ast
from analysis.topeft_run2 import plot_campaign_backend as backend
from analysis.topeft_run2 import plot_campaign_condor as condor


def test_public_control_and_worker_are_separate(tmp_path):
    directory = Path(condor.__file__).parent
    for name in ('condor_plotter_entry.sh', 'submit_plotter_condor.sh'):
        assert not (directory / name).exists()
    wrapper = (directory / 'run_plotter.sh').read_text()
    assert 'plot_campaign_manifest.py' in wrapper
    assert 'condor_submit' not in wrapper and 'make_cr_and_sr_plots.py' not in wrapper
    descriptor = {'argv': ['/fixture/python'], 'hard_limit_bytes': 1000, 'token': 'a' * 64}
    text = condor.submit_description(tmp_path / 'execution.json', descriptor)
    assert 'plot_campaign_backend' in text
    assert 'run_plotter.sh' not in text and 'plot_campaign_manifest' not in text
    for module in (condor, backend):
        source = Path(module.__file__).read_text()
        tree = ast.parse(source)
        names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        assert not {'run_unit', '_selected_view', 'read_manifest', 'prepare_campaign'} & names
        assert 'make_cr_and_sr_plots import' not in source
        assert 'plot_campaign_manifest' not in source
