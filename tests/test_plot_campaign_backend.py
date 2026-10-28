"""One-execution worker, explicit resource meanings and layout equivalence."""
import json
from pathlib import Path
import pytest
from analysis.topeft_run2 import plot_campaign_backend as backend
from topeft.modules import node_memory_guard as guard
from topeft.modules.plot_output_layout import canonical_observed_path, canonical_plot_destinations_for_period, normalize_plot_period


def test_worker_marker_prevents_second_renderer(tmp_path, monkeypatch):
    root = tmp_path / 'attempt_0001'; metadata = root / '_metadata'; metadata.mkdir(parents=True)
    argv = ['/fixture/python', str(backend.REPO_ROOT / 'analysis/topeft_run2/make_cr_and_sr_plots.py')]
    contract = dict(version=1, backend='condor', argv=argv, observer='cgroup_v2', hard_limit_bytes=100)
    descriptor = backend.execution_descriptor(argv, root, contract)
    path = metadata / 'execution.json'; path.write_text(json.dumps(descriptor))
    calls=[]
    monkeypatch.setattr(backend, 'run_guarded_child', lambda *a, **k: calls.append((a,k)) or (0,0,0))
    assert backend.main(['--worker', str(path)]) == 0
    with pytest.raises(FileExistsError): backend.main(['--worker', str(path)])
    assert len(calls) == 1
    assert calls[0][0][0] == argv
    assert calls[0][1]['observer'] == 'cgroup_v2'


@pytest.mark.parametrize('change', ['argv', 'token', 'root'])
def test_worker_rejects_changed_descriptor(tmp_path, monkeypatch, change):
    root = tmp_path / 'attempt_0001'; metadata=root/'_metadata'; metadata.mkdir(parents=True)
    argv=['/fixture/python', str(backend.REPO_ROOT/'analysis/topeft_run2/make_cr_and_sr_plots.py')]
    descriptor=backend.execution_descriptor(argv, root, dict(version=1,backend='condor',argv=argv,observer='cgroup_v2',hard_limit_bytes=100))
    if change=='argv': descriptor['argv']=['/fixture/python','run_plotter.sh']
    if change=='token': descriptor['token']='wrong'
    if change=='root': descriptor['attempt_root']=str(tmp_path/'elsewhere')
    path=metadata/'execution.json'; path.write_text(json.dumps(descriptor))
    monkeypatch.setattr(backend, 'run_guarded_child', lambda *a,**k: pytest.fail('child launched'))
    with pytest.raises(backend.CampaignError): backend.main(['--worker',str(path)])


@pytest.mark.parametrize('run,view', [('run2','combined'),('run2','UL16APV'),('run3','combined'),('run3','2023BPix')])
@pytest.mark.parametrize('variable', ['invmass','njets'])
def test_backend_neutral_observed_layout(tmp_path,run,view,variable):
    destinations=canonical_plot_destinations_for_period(tmp_path,normalize_plot_period(run,view),'category',f'category_{variable}_fitting.png',variable)
    for backend_name in ('local','condor'):
        # Backend execution identity never appears in a human destination.
        for path in destinations:
            assert canonical_observed_path(tmp_path,path.relative_to(tmp_path).as_posix()) == path
            assert backend_name not in path.relative_to(tmp_path).parts


@pytest.mark.parametrize('relative',['condor/run2/merged/cat/a.png','run2/merged_njets/cat/cat_invmass_fitting.png',
                                     '../run2/merged/cat/a.png','run2/merged/../a.png'])
def test_layout_rejects_noncanonical_namespace(tmp_path,relative):
    with pytest.raises(ValueError): canonical_observed_path(tmp_path,relative)


def cgroup_fixture(tmp_path,current='70',maximum='100'):
    root=tmp_path/'cgroup'; job=root/'allocation'; job.mkdir(parents=True)
    (job/'memory.current').write_text(current); (job/'memory.max').write_text(maximum)
    membership=tmp_path/'membership'; membership.write_text('0::/allocation\n')
    return root,membership


def test_cgroup_charge_has_explicit_semantics(tmp_path):
    root,membership=cgroup_fixture(tmp_path)
    sample=guard.read_cgroup_memory(root,membership)
    assert sample['cgroup_current_bytes']==70 and sample['cgroup_limit_bytes']==100
    assert 'pressure_used_bytes' not in sample and 'rss' not in sample


@pytest.mark.parametrize('current,maximum',[('70','max'),('-1','100'),('x','100'),('70','0')])
def test_invalid_or_unbounded_cgroup_has_no_node_fallback(tmp_path,monkeypatch,current,maximum):
    root,membership=cgroup_fixture(tmp_path,current,maximum)
    monkeypatch.setattr(guard,'read_node_pressure',lambda *a: pytest.fail('silent fallback'))
    with pytest.raises(ValueError): guard.read_cgroup_memory(root,membership)


def test_missing_cgroup_fails_before_child(tmp_path,monkeypatch):
    def unavailable(): raise OSError('allocation interface missing')
    monkeypatch.setattr(guard,'read_cgroup_memory',unavailable)
    monkeypatch.setattr(guard.subprocess,'Popen',lambda *a,**k: pytest.fail('child launched'))
    with pytest.raises(OSError):
        guard.run_guarded(['/bin/true'],hard_limit_bytes=100,evidence_path=tmp_path/'guard',observer='cgroup_v2')
    assert not (tmp_path/'guard').exists()


def test_cgroup_threshold_evidence_does_not_claim_node_pressure(tmp_path,monkeypatch):
    monkeypatch.setattr(guard,'read_cgroup_memory',lambda: dict(cgroup_current_bytes=70,cgroup_limit_bytes=100))
    result=guard.run_guarded(['/bin/sleep','0.2'],hard_limit_bytes=70,evidence_path=tmp_path/'guard',observer='cgroup_v2')
    assert result['termination_kind']=='threshold_killed'
    assert result['peak_cgroup_current_bytes']==70
    assert result['peak_pressure_used_bytes'] is None and result['minimum_mem_available_kb'] is None
    assert result['observer']=='cgroup_v2'


def test_probe_is_read_only_and_explicit(monkeypatch,capsys):
    monkeypatch.setattr(guard,'read_cgroup_memory',lambda: dict(cgroup_current_bytes=70,cgroup_limit_bytes=100))
    monkeypatch.setattr(guard.subprocess,'Popen',lambda *a,**k: pytest.fail('probe launched a child'))
    assert guard.main(['--probe','--observer','cgroup_v2'])==0
    assert json.loads(capsys.readouterr().out)['observer']=='cgroup_v2'


def test_exclusive_marker_syncs_directory_before_return(tmp_path, monkeypatch):
    import os
    import stat
    events=[]
    original=backend.os.fsync
    def sync(fd):
        events.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
        original(fd)
    monkeypatch.setattr(backend.os,'fsync',sync)
    backend.exclusive_json(tmp_path/'marker.json',{'token':'fixture'})
    assert events==['file','directory']
    with pytest.raises(FileExistsError): backend.exclusive_json(tmp_path/'marker.json',{})
    assert json.loads((tmp_path/'marker.json').read_text())=={'token':'fixture'}


def test_backend_missing_cgroup_is_definitely_not_started(tmp_path,monkeypatch):
    def unavailable(): raise OSError('missing cgroup')
    monkeypatch.setattr(guard,'read_cgroup_memory',unavailable)
    monkeypatch.setattr(backend.subprocess,'Popen',lambda *a,**k: pytest.fail('guard process launched'))
    with pytest.raises(backend.ChildDefinitelyNotStarted):
        backend.run_guarded_child(['/bin/true'],tmp_path,observer='cgroup_v2',hard_limit_bytes=100)
    assert not (tmp_path/'_metadata').exists()


def test_worker_directory_sync_failure_prevents_renderer(tmp_path,monkeypatch):
    import os
    import stat
    metadata=tmp_path/'_metadata';metadata.mkdir()
    argv=['/fixture/python',str(backend.REPO_ROOT/'analysis/topeft_run2/make_cr_and_sr_plots.py')]
    descriptor=backend.execution_descriptor(argv,tmp_path,dict(version=1,backend='condor',argv=argv,observer='cgroup_v2',hard_limit_bytes=100))
    path=metadata/'execution.json';path.write_text(json.dumps(descriptor))
    original=backend.os.fsync
    def sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode): raise OSError('directory sync failed')
        original(fd)
    monkeypatch.setattr(backend.os,'fsync',sync)
    monkeypatch.setattr(backend,'run_guarded_child',lambda *a,**k:pytest.fail('renderer launched'))
    with pytest.raises(OSError):backend.main(['--worker',str(path)])
    assert (metadata/'worker_started.json').exists()
