"""Adapter tests compose the maintained protocol with synthetic scheduler replies."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from analysis.topeft_run2 import event_scan_condor as condor
from analysis.topeft_run2 import event_scan as scan
from test_event_scan import unit_fixture


@pytest.fixture
def execution(unit_fixture,tmp_path,monkeypatch):
    unit,_=unit_fixture
    manifest=tmp_path/"units.json"
    manifest.write_text(json.dumps(dict(schema_version=1,units=[unit])))
    monkeypatch.setattr(condor.scheduler,"resolve_target",
                        lambda:dict(pool="collector.fixture.invalid",schedd="submit.fixture.invalid"))
    path=condor.prepare(manifest,unit["unit_id"],tmp_path/"attempt","/bin/true",sys.executable)
    return path,condor.read_descriptor(path)


def test_prepare_has_no_scheduler_effect_and_preserves_scientific_unit(execution,unit_fixture):
    path,descriptor=execution
    assert descriptor["unit"]==unit_fixture[0]
    assert not (path.parent/"submission_intent.json").exists()
    text=(path.parent/"execution.sub").read_text()
    assert text.count("queue 1\n")==1 and "max_retries = 0" in text
    assert "request_cpus = 1" in text and "allowed_execute_duration = 3600" in text
    assert "event_scan_condor --worker" in text
    assert "make_cr_and_sr_plots" not in text
    assert "executable = /usr/bin/true" in text or "executable = /bin/true" in text
    assert descriptor["token"] in text


def test_one_submit_reuses_identity_and_never_resubmits(execution,monkeypatch):
    path,descriptor=execution;calls=[]
    def submit(command,**kw):
        calls.append(command)
        intent=json.loads((path.parent/"submission_intent.json").read_text())
        assert intent["token"]==descriptor["token"]
        assert command[:5]==["condor_submit","-pool","collector.fixture.invalid","-name","submit.fixture.invalid"]
        return SimpleNamespace(returncode=0,stdout="123.0\n",stderr="")
    monkeypatch.setattr(condor.scheduler.subprocess,"run",submit)
    assert condor.submit(path)["job"]=="123.0"
    assert json.loads((path.parent/"scheduler_identity.json").read_text())["job"]=="123.0"
    with pytest.raises(ValueError,match="no resubmit"):condor.submit(path)
    assert len(calls)==1


def test_ambiguous_submit_keeps_intent_and_refuses_second_attempt(execution,monkeypatch):
    path,_=execution;calls=[]
    def timeout(command,**kw):
        calls.append(command)
        raise subprocess.TimeoutExpired(command,60)
    monkeypatch.setattr(condor.scheduler.subprocess,"run",timeout)
    with pytest.raises(condor.durable.ActiveChildUnresolved):condor.submit(path)
    assert (path.parent/"submission_intent.json").exists()
    assert not (path.parent/"scheduler_identity.json").exists()
    with pytest.raises(ValueError,match="no resubmit"):condor.submit(path)
    assert len(calls)==1


def test_worker_exclusive_marker_and_same_scanner(execution,monkeypatch):
    path,descriptor=execution;calls=[]
    monkeypatch.setattr(condor,"worker_identity",lambda token:"123.0")
    def run(unit,output,**kwargs):
        calls.append(unit);output.mkdir()
        (output/"receipt.json").write_text('{"status":"COMPLETE"}')
        return dict(status="COMPLETE")
    monkeypatch.setattr(scan,"run_unit",run)
    assert condor.worker(path)==0
    with pytest.raises(FileExistsError):condor.worker(path)
    assert calls==[descriptor["unit"]]
    assert json.loads((path.parent/"worker_receipt.json").read_text())["job"]=="123.0"


def test_worker_binds_exact_job_ad_token(execution,tmp_path,monkeypatch):
    _,descriptor=execution
    path=tmp_path/"job.ad"
    path.write_text('ClusterId = 42\nProcId = 0\nPlotExecutionToken = "'+descriptor["token"]+'"\n')
    monkeypatch.setenv("_CONDOR_JOB_AD",str(path))
    assert condor.worker_identity(descriptor["token"])=="42.0"
    with pytest.raises(ValueError,match="contradiction"):condor.worker_identity("0"*64)


def test_changed_manifest_or_descriptor_prevents_worker(execution,monkeypatch):
    path,descriptor=execution
    changed=dict(descriptor);changed["unit"]=dict(changed["unit"],entry_start=1)
    path.write_text(json.dumps(changed))
    monkeypatch.setattr(scan,"run_unit",lambda *a,**k:pytest.fail("science launched"))
    with pytest.raises(ValueError,match="descriptor changed"):condor.worker(path)


@pytest.fixture
def campaign(unit_fixture, tmp_path, monkeypatch):
    unit, _ = unit_fixture
    other = dict(unit, entry_start=7, entry_stop=9)
    other.pop('unit_id')
    other['unit_id'] = 'unit_' + condor.durable.identity(other)[:24]
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, units=[other, unit])))
    root = tmp_path / 'campaign'
    monkeypatch.setattr(condor.scheduler, 'resolve_target',
                        lambda: dict(pool='pool.fixture.invalid', schedd='schedd.fixture.invalid'))
    calls = []
    def submit_reply(command, **kwargs):
        assert command[0] == 'condor_submit'
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=f'{len(calls)}.0\n', stderr='')
    monkeypatch.setattr(condor.scheduler.subprocess, 'run', submit_reply)
    prepared = condor.prepare_manifest(manifest, root, '/bin/true', sys.executable)
    paths = [root / 'units' / uid / '_metadata/execution.json' for uid in prepared['unit_ids']]
    return root, manifest, prepared, paths, calls


def test_manifest_prepare_freezes_exact_bytes_and_reuses_canonical_attempts(campaign):
    root, manifest, prepared, paths, calls = campaign
    assert (root / 'manifest.json').read_bytes() == manifest.read_bytes()
    assert prepared['manifest_sha256'] == scan.native.sha256(manifest)
    assert prepared['unit_ids'] == sorted(prepared['unit_ids'])
    assert len(paths) == 2 and not calls
    for path in paths:
        descriptor = condor.read_descriptor(path)
        assert descriptor['manifest'] == str(root / 'manifest.json')
        assert (path.parent / 'execution.sub').read_text() == condor.submit_description(path, descriptor)
    assert condor.prepare_manifest(manifest, root, '/bin/true', sys.executable) == prepared
    assert sorted(p.name for p in root.iterdir()) == ['campaign.json', 'manifest.json', 'units']
    assert not list(root.rglob('*.sh'))


def test_campaign_root_required_in_cli(capsys):
    with pytest.raises(SystemExit) as error:
        condor.main(['prepare-manifest', '--manifest', 'm', '--wrapper', '/bin/true', '--python', sys.executable])
    assert error.value.code == 2
    assert '--campaign-root' in capsys.readouterr().err


@pytest.mark.parametrize('change', ['input', 'frozen', 'config', 'partial', 'collision'])
def test_campaign_identity_and_collision_fail_closed(campaign, change):
    root, manifest, _, paths, calls = campaign
    if change == 'input':
        manifest.write_text(manifest.read_text() + '\n')
    elif change == 'frozen':
        (root / 'manifest.json').write_text(manifest.read_text() + '\n')
    elif change == 'config':
        with pytest.raises(ValueError):
            condor.prepare_manifest(manifest, root, '/bin/true', sys.executable, max_runtime=20)
        return
    elif change == 'partial':
        (root / 'campaign.json').unlink()
    else:
        (root / 'units' / 'unexpected').mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        condor.prepare_manifest(manifest, root, '/bin/true', sys.executable)
    assert not calls


def test_manifest_submit_once_and_failed_worker_never_retried(campaign):
    root, _, prepared, paths, calls = campaign
    condor.submit_manifest(root)
    assert len(calls) == 2
    descriptor = condor.read_descriptor(paths[0])
    condor.durable.exclusive_json(paths[0].parent / 'worker_receipt.json',
        dict(token=descriptor['token'], job='1.0', status='STOPPED_NO_RETRY'))
    again = condor.submit_manifest(root)
    assert len(calls) == 2
    assert all(row['submission']['state'] == 'previously_submitted' for row in again['units'])


@pytest.mark.parametrize('artifact', ['campaign_submit_intent.json', 'submission_intent.json',
    'submission_response.json', 'scheduler_identity.json', 'worker_started.json', 'worker_receipt.json'])
def test_any_ambiguous_attempt_blocks_entire_campaign(campaign, artifact):
    root, _, _, paths, calls = campaign
    # Even ambiguity in the last unit blocks effects on the first.
    (paths[-1].parent / artifact).write_text('{}')
    with pytest.raises(ValueError, match='ambiguous'):
        condor.submit_manifest(root)
    assert calls == []


def test_manifest_reentry_after_interruption_only_submits_never_attempted(campaign, monkeypatch):
    root, _, _, _, calls = campaign
    original = condor.submit
    def interrupted(path):
        original(path)
        raise KeyboardInterrupt()
    monkeypatch.setattr(condor, 'submit', interrupted)
    with pytest.raises(KeyboardInterrupt):
        condor.submit_manifest(root)
    assert len(calls) == 1
    monkeypatch.setattr(condor, 'submit', original)
    condor.submit_manifest(root)
    assert len(calls) == 2


def test_manifest_interruption_before_submit_never_retries(campaign, monkeypatch):
    root, _, _, _, calls = campaign
    original = condor.submit
    monkeypatch.setattr(condor, 'submit', lambda path: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        condor.submit_manifest(root)
    monkeypatch.setattr(condor, 'submit', original)
    with pytest.raises(ValueError, match='ambiguous'):
        condor.submit_manifest(root)
    assert calls == []


@pytest.mark.parametrize('change', ['response', 'intent', 'worker', 'claim'])
def test_submitted_contradictions_are_not_silently_skipped(campaign, change):
    root, _, _, paths, calls = campaign
    condor.submit_manifest(root)
    metadata = paths[0].parent
    if change == 'response':
        (metadata / 'submission_response.json').write_text('{"returncode":0,"stdout":"999.0"}')
    elif change == 'intent':
        value = json.loads((metadata / 'submission_intent.json').read_text())
        value['token'] = 'wrong'
        (metadata / 'submission_intent.json').write_text(json.dumps(value))
    elif change == 'worker':
        (metadata / 'worker_receipt.json').write_text('{"token":"wrong","job":"1.0"}')
    else:
        (metadata / 'campaign_submit_intent.json').write_text('{}')
    with pytest.raises((ValueError, condor.durable.ActiveChildUnresolved)):
        condor.submit_manifest(root)
    assert len(calls) == 2


def test_status_manifest_preserves_distinct_states_and_has_no_submit(campaign, monkeypatch):
    root, _, _, paths, calls = campaign
    condor.submit_manifest(root)
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    states = iter([dict(state='pending', ad={'JobStatus': 2}), dict(state='terminal_failure', ad={'ExitCode': 1})])
    def observe(path, descriptor, *, read_only):
        assert read_only is True
        return next(states)
    monkeypatch.setattr(condor.scheduler, 'observe', observe)
    result = condor.status_manifest(root)
    assert [u['scheduler']['state'] for u in result['units']] == ['pending', 'terminal_failure']
    assert len(calls) == 2
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_plan_manifest_is_zero_effect_and_structural(campaign, monkeypatch):
    root, manifest, prepared, _, calls = campaign
    monkeypatch.setattr(scan, 'validate_unit', lambda *a, **k: pytest.fail('input/science probe'))
    monkeypatch.setattr(scan, 'source_identity', lambda *a: pytest.fail('ROOT read'))
    monkeypatch.setattr(condor, 'prepare', lambda *a, **k: pytest.fail('prepare'))
    before = {str(p): p.read_bytes() for p in manifest.parent.rglob('*') if p.is_file()}
    plan = condor.plan_manifest(manifest)
    assert plan['unit_count'] == 2
    assert [u['unit_id'] for u in plan['units']] == prepared['unit_ids']
    assert not calls
    assert before == {str(p): p.read_bytes() for p in manifest.parent.rglob('*') if p.is_file()}


def test_manifest_concurrent_controller_fails_before_submit(campaign):
    import fcntl
    root, _, _, _, calls = campaign
    with (root / 'campaign.json').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            condor.submit_manifest(root)
    assert not calls


def test_partial_preparation_cannot_be_silently_resumed(campaign, monkeypatch):
    root, manifest, _, _, calls = campaign
    original = condor.prepare
    interrupted_root = root.parent / 'partial'
    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt()
    monkeypatch.setattr(condor, 'prepare', interrupted)
    with pytest.raises(KeyboardInterrupt):
        condor.prepare_manifest(manifest, interrupted_root, '/bin/true', sys.executable)
    monkeypatch.setattr(condor, 'prepare', original)
    with pytest.raises(ValueError):
        condor.prepare_manifest(manifest, interrupted_root, '/bin/true', sys.executable)
    assert not calls


@pytest.mark.parametrize('state', ['running', 'terminal_success', 'terminal_failure', 'unresolved'])
def test_manifest_status_uses_canonical_observer_read_only(campaign, monkeypatch, state):
    root, _, _, paths, calls = campaign
    condor.submit_manifest(root)
    if state == 'unresolved':
        (paths[0].parent / 'scheduler_identity.json').unlink()
        monkeypatch.setattr(condor.scheduler, '_recover', lambda *a: '1.0')
    status = dict(running=2, terminal_success=4, terminal_failure=4, unresolved=0)[state]
    monkeypatch.setattr(condor.scheduler, '_query', lambda job, command, intent:
        dict(JobStatus=status, ExitCode=int(state == 'terminal_failure'), ExitBySignal=False, NumJobStarts=1))
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = condor.status_manifest(root)
    expected = 'pending' if state == 'running' else state
    assert all(u['scheduler']['state'] == expected for u in result['units'])
    assert len(calls) == 2
    assert before == {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_status_reports_ambiguous_worker_receipt_without_dropping_units(campaign, monkeypatch):
    root, _, _, paths, calls = campaign
    condor.submit_manifest(root)
    (paths[0].parent / 'worker_receipt.json').write_text('{')
    monkeypatch.setattr(condor.scheduler, 'observe', lambda *a, **k: dict(state='pending'))
    result = condor.status_manifest(root)
    assert len(result['units']) == 2
    assert result['units'][0]['local_submission']['state'] == 'unresolved'
    assert result['units'][0]['worker_receipt']['state'] == 'unresolved'
    assert len(calls) == 2


def test_campaign_reentry_checks_prior_submit_descriptions(campaign):
    root, _, _, paths, calls = campaign
    condor.submit_manifest(root)
    (paths[-1].parent / 'execution.sub').write_text('changed')
    with pytest.raises(ValueError, match='submit description changed'):
        condor.submit_manifest(root)
    assert len(calls) == 2
