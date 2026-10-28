"""Synthetic per-unit scheduler protocol; never runs an HTCondor command."""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from analysis.topeft_run2 import plot_campaign_backend as backend
from analysis.topeft_run2 import plot_campaign_condor as condor


@pytest.fixture
def execution(tmp_path):
    root = tmp_path / 'unit' / 'attempt_0001'
    root.mkdir(parents=True)
    argv = ['/fixture/python', str(backend.REPO_ROOT / 'analysis/topeft_run2/make_cr_and_sr_plots.py'),
            '-f', '/unopened/input', '--sr']
    contract = dict(version=1, backend='condor', argv=argv, observer='cgroup_v2', hard_limit_bytes=1000000)
    descriptor = backend.execution_descriptor(argv, root, contract)
    metadata = root / '_metadata'
    metadata.mkdir()
    path = metadata / 'execution.json'
    path.write_text(json.dumps(descriptor))
    return path, descriptor


def response(stdout, returncode=0):
    return SimpleNamespace(stdout=stdout, stderr='fixture', returncode=returncode)


def test_descriptor_one_direct_worker_no_control_plane(execution):
    path, descriptor = execution
    text = condor.submit_description(path, descriptor)
    assert text.count('queue 1') == 1
    assert 'plot_campaign_backend' in text
    assert 'run_plotter.sh' not in text and 'plot_campaign_manifest' not in text
    assert 'NumJobStarts > 1' in text
    assert descriptor['token'] in text
    assert 'request_cpus = 1' in text and 'should_transfer_files = NO' in text


@pytest.mark.parametrize('stdout', ['123.0\n', '123.0 - 123.0\n'])
def test_submit_records_identity_once(execution, monkeypatch, stdout):
    path, descriptor = execution
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        assert (path.parent / 'submission_intent.json').is_file()
        return response(stdout)
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    assert condor.submit(path, descriptor)['job'] == '123.0'
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.submit(path, descriptor)
    assert len(calls) == 1
    assert json.loads((path.parent / 'scheduler_identity.json').read_text())['token'] == descriptor['token']


@pytest.mark.parametrize('stdout,code', [('', 0), ('123.0 - 123.1', 0), ('123.0\n456.0', 0),
                                       ('123.0', 1), ('', 1), ('malformed', 0)])
def test_submit_nonzero_or_ambiguous_never_proves_failure(execution, monkeypatch, stdout, code):
    path, descriptor = execution
    monkeypatch.setattr(condor.subprocess, 'run', lambda *a, **k: response(stdout, code))
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.submit(path, descriptor)
    assert not (path.parent / 'scheduler_identity.json').exists()
    assert (path.parent / 'submission_intent.json').exists()


def test_submit_timeout_is_unknown(execution, monkeypatch):
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired('fake_submit', 60)
    monkeypatch.setattr(condor.subprocess, 'run', timeout)
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.submit(*execution)


def bind(execution):
    path, descriptor = execution
    (path.parent / 'scheduler_identity.json').write_text(json.dumps({'job':'123.0','token':descriptor['token']}))
    return dict(ClusterId=123, ProcId=0, PlotExecutionToken=descriptor['token'], JobStatus=4,
                ExitCode=0, ExitBySignal=False, NumJobStarts=1)


@pytest.mark.parametrize('fields,state', [({}, 'terminal_success'), ({'ExitCode':7}, 'terminal_failure'),
    ({'ExitBySignal':True}, 'terminal_failure'), ({'NumJobStarts':2}, 'terminal_failure'),
    ({'ExitCode':None}, 'unresolved'), ({'NumJobStarts':None}, 'unresolved'),
    ({'JobStatus':3}, 'terminal_failure'), ({'JobStatus':5}, 'pending'),
    ({'JobStatus':2}, 'pending'), ({'JobStatus':99}, 'unresolved')])
def test_query_terminal_and_pending(execution, monkeypatch, fields, state):
    ad = {**bind(execution), **fields}
    monkeypatch.setattr(condor.subprocess, 'run', lambda *a, **k: response(json.dumps([ad])))
    assert condor.observe(*execution)['state'] == state


def test_query_history_and_unknown_do_not_resubmit(execution, monkeypatch):
    ad = bind(execution); calls = []
    def fake(command, **kwargs):
        calls.append(command[0])
        return response(json.dumps([] if command[0] == 'condor_q' else [ad]))
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    assert condor.observe(*execution)['state'] == 'terminal_success'
    assert calls == ['condor_q', 'condor_history']
    monkeypatch.setattr(condor.subprocess, 'run', lambda *a, **k: response('[]'))
    assert condor.observe(*execution)['state'] == 'unresolved'


@pytest.mark.parametrize('corruption', ['job', 'token', 'duplicate', 'json', 'error'])
def test_query_contradictions_fail_closed(execution, monkeypatch, corruption):
    ad = bind(execution)
    if corruption == 'job': ad['ProcId'] = 1
    if corruption == 'token': ad['PlotExecutionToken'] = 'wrong'
    body = json.dumps([ad, ad] if corruption == 'duplicate' else [ad])
    if corruption == 'json': body = 'invalid'
    monkeypatch.setattr(condor.subprocess, 'run', lambda *a, **k: response(body, int(corruption == 'error')))
    with pytest.raises((backend.ActiveChildUnresolved, ValueError)):
        condor.observe(*execution)


def test_missing_identity_never_queries_or_submits(execution, monkeypatch):
    monkeypatch.setattr(condor.subprocess, 'run', lambda *a, **k: pytest.fail('scheduler must not be called'))
    assert condor.observe(*execution)['state'] == 'unresolved'


def test_intent_directory_sync_failure_never_submits(execution,monkeypatch):
    import os
    import stat
    original=backend.os.fsync
    def sync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode): raise OSError('directory sync failed')
        original(fd)
    monkeypatch.setattr(backend.os,'fsync',sync)
    monkeypatch.setattr(condor.subprocess,'run',lambda *a,**k:pytest.fail('submit after failed durable intent'))
    with pytest.raises(backend.ActiveChildUnresolved):condor.submit(*execution)
    assert (execution[0].parent/'submission_intent.json').exists()
