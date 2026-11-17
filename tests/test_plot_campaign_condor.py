"""Synthetic per-unit scheduler protocol; never runs an HTCondor command."""
import json
import os
import pwd
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from analysis.topeft_run2 import plot_campaign_backend as backend
from analysis.topeft_run2 import plot_campaign_condor as condor

fixture_pool = 'collector.fixture.invalid'
fixture_schedd = 'submit.fixture.invalid'
real_resolve_target = condor.resolve_target


@pytest.fixture(autouse=True)
def fake_target(monkeypatch):
    monkeypatch.setattr(condor, 'resolve_target', lambda: {'pool': fixture_pool, 'schedd': fixture_schedd})


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


def test_submit_filename_syntax_keeps_string_fields_and_resources(execution):
    path, descriptor = execution
    descriptor['hard_limit_bytes'] = 192000000000
    text = condor.submit_description(path, descriptor)
    for key, value in {
        'executable': descriptor['argv'][0], 'initialdir': str(backend.REPO_ROOT),
        'output': str(path.parent / 'worker.out'), 'error': str(path.parent / 'worker.err'),
        'log': str(path.parent / 'scheduler.log'),
    }.items():
        assert f'{key} = {value}\n' in text
        assert f'{key} = "{value}"' not in text
    assert f'environment = "PYTHONPATH={backend.REPO_ROOT}"\n' in text
    assert f'+PlotExecutionToken = "{descriptor["token"]}"\n' in text
    assert f"arguments = \"'-m' 'analysis.topeft_run2.plot_campaign_backend' '--worker' '{path}'\"\n" in text
    for line in ('request_cpus = 1', 'request_memory = 183106', 'request_disk = 10240',
                 '+MaxRuntime = 86400', 'queue 1'):
        assert text.count(line + '\n') == 1


@pytest.mark.parametrize('unsafe', ['/tmp/space in name', '/tmp/bad\nqueue 2', '/tmp/"quote'])
def test_submit_rejects_unsafe_filename(execution, unsafe):
    path, descriptor = execution
    descriptor['argv'][0] = unsafe
    with pytest.raises(ValueError, match='Unsafe Condor filename'):
        condor.submit_description(path, descriptor)


def rejected_execution(execution):
    path, descriptor = execution
    metadata = path.parent
    root = str(backend.REPO_ROOT)
    legacy = condor.submit_description(path, descriptor)
    for key in ('executable', 'initialdir', 'output', 'error', 'log'):
        line = next(line for line in legacy.splitlines() if line.startswith(key + ' = '))
        value = line.split(' = ', 1)[1]
        legacy = legacy.replace(line + '\n', f'{key} = "{value}"\n')
    (metadata / 'execution.sub').write_text(legacy)
    (metadata / 'submission_intent.json').write_text(json.dumps({
        'token': descriptor['token'], 'pool': fixture_pool, 'schedd': fixture_schedd}))
    (metadata / 'submission_response.json').write_text(json.dumps({
        'returncode': 1, 'stdout': '',
        'stderr': f'ERROR: No such directory: {root}/"{root}"\n'}))
    return path, descriptor


def test_rejected_submit_reuses_attempt_and_preserves_evidence(execution, monkeypatch):
    path, descriptor = rejected_execution(execution)
    original = {name: (path.parent / name).read_bytes()
                for name in ('execution.sub', 'submission_response.json')}
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        assert command[1:5] == ['-pool', fixture_pool, '-name', fixture_schedd]
        if command[0] == 'condor_submit':
            assert (path.parent / 'recovery_submit_intent.json').is_file()
            for name, content in original.items():
                assert (path.parent / ('rejected_' + name)).read_bytes() == content
            assert command[-1] == str(path.parent / 'recovery_execution.sub')
            return response('123.0\n')
        return response('[]')
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    assert condor.recover_rejected_submit(path, descriptor)['job'] == '123.0'
    assert [call[0] for call in calls] == ['condor_q', 'condor_history', 'condor_submit']
    assert json.loads((path.parent / 'scheduler_identity.json').read_text()) == {
        'token': descriptor['token'], 'pool': fixture_pool, 'schedd': fixture_schedd, 'job': '123.0'}
    provenance = json.loads((path.parent / 'rejected_submission_provenance.json').read_text())
    for name, content in original.items():
        assert (path.parent / name).read_bytes() == content
        assert provenance['sha256'][name] == backend.sha256_file(path.parent / name)
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.recover_rejected_submit(path, descriptor)
    assert len(calls) == 3


@pytest.mark.parametrize('reason', ['wrong_response', 'missing_response', 'queue_match',
                                    'history_match', 'identity', 'ambiguous_submit'])
def test_rejected_submit_fail_closed(execution, monkeypatch, reason):
    path, descriptor = rejected_execution(execution)
    if reason == 'wrong_response':
        (path.parent / 'submission_response.json').write_text(json.dumps({
            'returncode': 1, 'stdout': '', 'stderr': 'unrelated failure'}))
    elif reason == 'missing_response':
        (path.parent / 'submission_response.json').unlink()
    elif reason == 'identity':
        (path.parent / 'scheduler_identity.json').write_text('{}')
    calls = []
    def fake(command, **kwargs):
        calls.append(command[0])
        if command[0] == 'condor_submit':
            assert reason == 'ambiguous_submit'
            return response('', 1)
        if reason == 'queue_match' or (reason == 'history_match' and command[0] == 'condor_history'):
            ad = {'ClusterId': 123, 'ProcId': 0, 'Owner': pwd.getpwuid(os.getuid()).pw_name,
                  'PlotExecutionToken': descriptor['token']}
            return response(json.dumps([ad]))
        return response('[]')
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.recover_rejected_submit(path, descriptor)
    assert calls.count('condor_submit') == int(reason == 'ambiguous_submit')
    if reason == 'ambiguous_submit':
        with pytest.raises(backend.ActiveChildUnresolved):
            condor.recover_rejected_submit(path, descriptor)
        assert calls.count('condor_submit') == 1


@pytest.mark.parametrize('stdout', ['123.0\n', '123.0 - 123.0\n'])
def test_submit_records_identity_once(execution, monkeypatch, stdout):
    path, descriptor = execution
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        intent = json.loads((path.parent / 'submission_intent.json').read_text())
        assert intent == {'token': descriptor['token'], 'pool': fixture_pool, 'schedd': fixture_schedd}
        assert command == ['condor_submit', '-pool', fixture_pool, '-name', fixture_schedd,
                           '-terse', str(path.parent / 'execution.sub')]
        return response(stdout)
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    assert condor.submit(path, descriptor)['job'] == '123.0'
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.submit(path, descriptor)
    assert len(calls) == 1
    assert json.loads((path.parent / 'scheduler_identity.json').read_text()) == {
        'token': descriptor['token'], 'pool': fixture_pool, 'schedd': fixture_schedd, 'job': '123.0'}


def test_intent_directory_sync_precedes_submit(execution, monkeypatch):
    events = []
    original_fsync = backend.os.fsync
    def sync(fd):
        events.append('directory_sync' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file_sync')
        return original_fsync(fd)
    def fake(command, **kwargs):
        assert command[0] == 'condor_submit'
        assert json.loads((execution[0].parent / 'submission_intent.json').read_text()) == {
            'token': execution[1]['token'], 'pool': fixture_pool, 'schedd': fixture_schedd}
        assert events[:2] == ['file_sync', 'directory_sync']
        events.append('submit')
        return response('123.0\n')
    monkeypatch.setattr(backend.os, 'fsync', sync)
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    assert condor.submit(*execution)['job'] == '123.0'
    assert events.index('directory_sync') < events.index('submit')


@pytest.mark.parametrize('stdout,code', [('', 0), ('123.0 - 123.1', 0), ('123.0\n456.0', 0),
                                       ('123.0', 1), ('', 1), ('malformed', 0)])
def test_submit_nonzero_or_ambiguous_never_proves_failure(execution, monkeypatch, stdout, code):
    path, descriptor = execution
    monkeypatch.setattr(condor.subprocess, 'run', lambda *a, **k: response(stdout, code))
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.submit(path, descriptor)
    assert not (path.parent / 'scheduler_identity.json').exists()
    assert json.loads((path.parent / 'submission_intent.json').read_text()) == {
        'token': descriptor['token'], 'pool': fixture_pool, 'schedd': fixture_schedd}


def test_submit_timeout_is_unknown(execution, monkeypatch):
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired('fake_submit', 60)
    monkeypatch.setattr(condor.subprocess, 'run', timeout)
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.submit(*execution)
    path, descriptor = execution
    assert json.loads((path.parent / 'submission_intent.json').read_text()) == {
        'token': descriptor['token'], 'pool': fixture_pool, 'schedd': fixture_schedd}
    assert not (path.parent / 'scheduler_identity.json').exists()


def test_submit_spawn_oserror_is_definitely_not_started(execution, monkeypatch):
    path, descriptor = execution
    calls = []
    def failed_spawn(command, **kwargs):
        calls.append(command)
        assert command == ['condor_submit', '-pool', fixture_pool, '-name', fixture_schedd,
                           '-terse', str(path.parent / 'execution.sub')]
        assert json.loads((path.parent / 'submission_intent.json').read_text()) == {
            'token': descriptor['token'], 'pool': fixture_pool, 'schedd': fixture_schedd}
        raise OSError('fake local spawn failure')
    monkeypatch.setattr(condor.subprocess, 'run', failed_spawn)
    with pytest.raises(backend.ChildDefinitelyNotStarted):
        condor.submit(path, descriptor)
    assert len(calls) == 1
    assert not (path.parent / 'scheduler_identity.json').exists()
    assert not (path.parent / 'submission_response.json').exists()
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.submit(path, descriptor)
    assert len(calls) == 1


def test_response_write_failure_after_submit_stays_unresolved(execution, monkeypatch):
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        return response('123.0\n')
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    def failed_response(path, value):
        raise OSError('fake response write failure')
    monkeypatch.setattr(backend, 'atomic_json', failed_response)
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.submit(*execution)
    assert len(calls) == 1
    assert json.loads((execution[0].parent / 'submission_intent.json').read_text()) == {
        'token': execution[1]['token'], 'pool': fixture_pool, 'schedd': fixture_schedd}
    assert not (execution[0].parent / 'scheduler_identity.json').exists()
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.submit(*execution)
    assert len(calls) == 1


def bind(execution):
    path, descriptor = execution
    intent = {'token': descriptor['token'], 'pool': fixture_pool, 'schedd': fixture_schedd}
    (path.parent / 'submission_intent.json').write_text(json.dumps(intent))
    (path.parent / 'scheduler_identity.json').write_text(json.dumps({**intent, 'job':'123.0'}))
    return dict(ClusterId=123, ProcId=0, PlotExecutionToken=descriptor['token'], JobStatus=4,
                Owner=pwd.getpwuid(os.getuid()).pw_name, ExitCode=0, ExitBySignal=False,
                NumJobStarts=1)


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
        assert command[1:5] == ['-pool', fixture_pool, '-name', fixture_schedd]
        return response('' if command[0] == 'condor_q' else json.dumps([ad]))
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    assert condor.observe(*execution)['state'] == 'terminal_success'
    assert calls == ['condor_q', 'condor_history']
    monkeypatch.setattr(condor.subprocess, 'run', lambda *a, **k: response('[]'))
    assert condor.observe(*execution)['state'] == 'unresolved'


@pytest.mark.parametrize('in_history', [False, True])
def test_read_only_observation_preserves_raw_metrics_and_files(execution, monkeypatch, in_history):
    ad = {**bind(execution), 'JobStatus': 5, 'HoldReason': 'memory exceeded',
          'MemoryUsage': 0, 'ResidentSetSize': 123456, 'ExitSignal': 9}
    path, descriptor = execution
    # Read-only recovery may find the job but must not persist its identity.
    (path.parent / 'scheduler_identity.json').unlink()
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        assert command[0] in {'condor_q', 'condor_history'}
        if '-constraint' not in command:
            attrs = command[command.index('-attributes') + 1].split(',')
            assert {'HoldReason', 'MemoryUsage', 'ResidentSetSize', 'ExitSignal'} <= set(attrs)
        return response('[]' if in_history and command[0] == 'condor_q' else json.dumps([ad]))
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    observation = condor.observe(path, descriptor, read_only=True)
    assert observation['ad'] == ad
    assert observation['source'] == ('history' if in_history else 'queue')
    assert observation['state'] == 'pending'
    assert {p.name: p.read_bytes() for p in path.parent.iterdir()} == before
    assert len(calls) == (4 if in_history else 2)


def test_retired_monitor_direct_invocation_has_migration_error(tmp_path):
    import sys
    result = subprocess.run([sys.executable, condor.__file__, 'monitor',
                             '--diagnostics-dir', str(tmp_path)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'run_plotter.sh --status' in result.stderr
    assert '--observe-scheduler' in result.stderr
    assert not list(tmp_path.iterdir())


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


@pytest.mark.parametrize('configured,ads,valid', [
    (None, [{'Name': fixture_schedd, 'Machine': 'local.fixture.invalid'}], True),
    (fixture_schedd, [{'Name': fixture_schedd, 'Machine': 'local.fixture.invalid'}], True),
    (None, [], False),
    (None, [{'Name': fixture_schedd, 'Machine': 'local.fixture.invalid'}] * 2, False),
    (fixture_schedd, [{'Name': 'other.fixture.invalid', 'Machine': 'local.fixture.invalid'}], False),
])
def test_resolver_uses_config_and_exact_local_ad(monkeypatch, configured, ads, valid):
    calls = []
    monkeypatch.setattr(condor, 'resolve_target', real_resolve_target)
    def fake(command, **kwargs):
        calls.append(command)
        if command[0] == 'condor_config_val':
            values = {'COLLECTOR_HOST': fixture_pool, 'FULL_HOSTNAME': 'local.fixture.invalid',
                      'SCHEDD_NAME': configured}
            value = values[command[1]]
            return response(value + '\n') if value is not None else \
                SimpleNamespace(stdout='', stderr='Not defined: SCHEDD_NAME', returncode=1)
        assert command[:5] == ['condor_status', '-pool', fixture_pool, '-schedd', '-constraint']
        expected = ('Name == ' + json.dumps(fixture_schedd) if configured else
                    'Machine == ' + json.dumps('local.fixture.invalid'))
        assert command[5] == expected
        return response(json.dumps(ads))
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    if valid:
        assert condor.resolve_target() == {'pool': fixture_pool, 'schedd': fixture_schedd}
    else:
        with pytest.raises(backend.ChildDefinitelyNotStarted):
            condor.resolve_target()
    assert len(calls) == 4


@pytest.mark.parametrize('pool', ['', 'one.fixture.invalid,two.fixture.invalid',
                                  'one.fixture.invalid two.fixture.invalid'])
def test_ambiguous_pool_fails_before_submit(execution, monkeypatch, pool):
    monkeypatch.setattr(condor, 'resolve_target', lambda: condor._target_value(pool, 'pool'))
    monkeypatch.setattr(condor.subprocess, 'run', lambda *a, **k: pytest.fail('scheduler called'))
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.submit(*execution)
    assert not (execution[0].parent / 'submission_intent.json').exists()


@pytest.mark.parametrize('source', ['condor_q', 'condor_history'])
def test_ambiguous_submit_recovery_is_bound_and_never_submits(execution, monkeypatch, source):
    path, descriptor = execution
    intent = {'token': descriptor['token'], 'pool': fixture_pool, 'schedd': fixture_schedd}
    (path.parent / 'submission_intent.json').write_text(json.dumps(intent))
    ad = dict(ClusterId=123, ProcId=0, PlotExecutionToken=descriptor['token'],
              Owner=pwd.getpwuid(os.getuid()).pw_name, JobStatus=2)
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        assert command[0] in ('condor_q', 'condor_history')
        assert command[1:5] == ['-pool', fixture_pool, '-name', fixture_schedd]
        if '-constraint' in command:
            constraint = command[command.index('-constraint') + 1]
            assert constraint == ('PlotExecutionToken == ' + json.dumps(descriptor['token']) +
                                  ' && Owner == ' + json.dumps(ad['Owner']))
            assert command[command.index('-limit') + 1] == '2'
            return response(json.dumps([ad]) if command[0] == source else '')
        assert command[5] == '123.0'
        return response(json.dumps([ad]))
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    monkeypatch.setattr(condor, 'resolve_target', lambda: pytest.fail('resolver called on resume'))
    assert condor.observe(path, descriptor)['state'] == 'pending'
    assert json.loads((path.parent / 'scheduler_identity.json').read_text()) == {**intent, 'job': '123.0'}
    assert [call[0] for call in calls] == (['condor_q', 'condor_q'] if source == 'condor_q'
                                           else ['condor_q', 'condor_history', 'condor_q'])


@pytest.mark.parametrize('ads', [[], [0, 1]])
def test_recovery_zero_or_multiple_never_submits(execution, monkeypatch, ads):
    path, descriptor = execution
    (path.parent / 'submission_intent.json').write_text(json.dumps({
        'token': descriptor['token'], 'pool': fixture_pool, 'schedd': fixture_schedd}))
    ad = dict(ClusterId=123, ProcId=0, PlotExecutionToken=descriptor['token'],
              Owner=pwd.getpwuid(os.getuid()).pw_name, JobStatus=2)
    calls = []
    def fake(command, **kwargs):
        calls.append(command[0])
        assert command[0] in ('condor_q', 'condor_history')
        return response(json.dumps([{**ad, 'ClusterId': 123 + i} for i in ads]))
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    if ads:
        with pytest.raises(backend.ActiveChildUnresolved):
            condor.observe(path, descriptor)
        assert calls == ['condor_q']
    else:
        assert condor.observe(path, descriptor)['state'] == 'unresolved'
        assert calls == ['condor_q', 'condor_history']
    assert not (path.parent / 'scheduler_identity.json').exists()


@pytest.mark.parametrize('field,value', [('pool', 'other.fixture.invalid'),
                                          ('pool', '-wrong-option'),
                                          ('schedd', 'other.fixture.invalid'),
                                          ('token', 'wrong'), ('job', '123.1')])
def test_identity_contradiction_prevents_query(execution, monkeypatch, field, value):
    bind(execution)
    path, _ = execution
    identity_path = path.parent / 'scheduler_identity.json'
    identity = json.loads(identity_path.read_text())
    identity[field] = value
    identity_path.write_text(json.dumps(identity))
    monkeypatch.setattr(condor.subprocess, 'run', lambda *a, **k: pytest.fail('query after contradiction'))
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.observe(*execution)


def test_existing_binding_ignores_changed_default(execution, monkeypatch):
    ad = bind(execution)
    monkeypatch.setattr(condor, 'resolve_target', lambda: pytest.fail('resolver called on resume'))
    def fake(command, **kwargs):
        assert command[1:5] == ['-pool', fixture_pool, '-name', fixture_schedd]
        return response(json.dumps([ad]))
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    assert condor.observe(*execution)['state'] == 'terminal_success'


@pytest.mark.parametrize('field,value', [('Owner', 'other_owner'), ('ProcId', 1),
                                          ('PlotExecutionToken', 'wrong')])
def test_recovery_rejects_wrong_ad_identity(execution, monkeypatch, field, value):
    path, descriptor = execution
    (path.parent / 'submission_intent.json').write_text(json.dumps({
        'token': descriptor['token'], 'pool': fixture_pool, 'schedd': fixture_schedd}))
    ad = dict(ClusterId=123, ProcId=0, PlotExecutionToken=descriptor['token'],
              Owner=pwd.getpwuid(os.getuid()).pw_name, JobStatus=2)
    ad[field] = value
    def fake(command, **kwargs):
        assert command[0] == 'condor_q'
        return response(json.dumps([ad]))
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.observe(*execution)
    assert not (path.parent / 'scheduler_identity.json').exists()


def test_exact_history_query_has_one_ad_limit_and_short_timeout(execution, monkeypatch):
    path, descriptor = execution
    intent = {'token': descriptor['token'], 'pool': fixture_pool, 'schedd': fixture_schedd}
    (path.parent / 'submission_intent.json').write_text(json.dumps(intent))
    (path.parent / 'scheduler_identity.json').write_text(json.dumps({**intent, 'job': '123.0'}))
    owner = pwd.getpwuid(os.getuid()).pw_name
    ad = dict(ClusterId=123, ProcId=0, PlotExecutionToken=descriptor['token'],
              Owner=owner, JobStatus=4, ExitCode=0, ExitBySignal=False, NumJobStarts=1)
    calls = []
    def fake(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == 'condor_q':
            return response('[]')
        return response(json.dumps([ad]))
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    assert condor.observe(path, descriptor)['state'] == 'terminal_success'
    history, options = calls[-1]
    assert history[:6] == ['condor_history', '-pool', fixture_pool, '-name', fixture_schedd, '123.0']
    assert history[history.index('-limit') + 1] == '1'
    assert options['timeout'] == 10


def test_history_timeout_is_unresolved_and_never_submits(execution, monkeypatch):
    path, descriptor = execution
    intent = {'token': descriptor['token'], 'pool': fixture_pool, 'schedd': fixture_schedd}
    (path.parent / 'submission_intent.json').write_text(json.dumps(intent))
    (path.parent / 'scheduler_identity.json').write_text(json.dumps({**intent, 'job': '123.0'}))
    calls = []
    def fake(command, **kwargs):
        calls.append(command[0])
        if command[0] == 'condor_history':
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        return response('[]')
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.observe(path, descriptor)
    assert calls == ['condor_q', 'condor_history']


def test_old_token_only_intent_never_resolves_or_queries(execution, monkeypatch):
    path, descriptor = execution
    (path.parent / 'submission_intent.json').write_text(json.dumps({'token': descriptor['token']}))
    monkeypatch.setattr(condor, 'resolve_target', lambda: pytest.fail('resolver called on old intent'))
    monkeypatch.setattr(condor.subprocess, 'run', lambda *a, **k: pytest.fail('scheduler queried'))
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.observe(*execution)


def test_occupied_submit_file_without_intent_is_unresolved(execution, monkeypatch):
    path, _ = execution
    (path.parent / 'execution.sub').write_text('another setup')
    monkeypatch.setattr(condor.subprocess, 'run', lambda *a, **k: pytest.fail('scheduler called'))
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.submit(*execution)
    assert not (path.parent / 'submission_intent.json').exists()


def test_orphan_identity_prevents_any_new_submit(execution, monkeypatch):
    path, descriptor = execution
    (path.parent / 'scheduler_identity.json').write_text(json.dumps({
        'token': descriptor['token'], 'job': '123.0'}))
    monkeypatch.setattr(condor, 'resolve_target', lambda: pytest.fail('resolver called'))
    monkeypatch.setattr(condor.subprocess, 'run', lambda *a, **k: pytest.fail('scheduler called'))
    with pytest.raises(backend.ActiveChildUnresolved):
        condor.submit(*execution)
    assert not (path.parent / 'submission_intent.json').exists()


@pytest.mark.parametrize('pool', ['', 'one.fixture.invalid,two.fixture.invalid'])
def test_bad_configured_pool_fails_before_intent(execution, monkeypatch, pool):
    monkeypatch.setattr(condor, 'resolve_target', real_resolve_target)
    def fake(command, **kwargs):
        assert command[0] == 'condor_config_val'
        values = {'COLLECTOR_HOST': pool, 'FULL_HOSTNAME': 'local.fixture.invalid',
                  'SCHEDD_NAME': 'submit.fixture.invalid'}
        return response(values[command[1]] + '\n')
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    with pytest.raises(backend.ChildDefinitelyNotStarted):
        condor.submit(*execution)
    assert not (execution[0].parent / 'submission_intent.json').exists()


def test_unreachable_schedd_lookup_fails_before_intent(execution, monkeypatch):
    monkeypatch.setattr(condor, 'resolve_target', real_resolve_target)
    def fake(command, **kwargs):
        if command[0] == 'condor_status':
            raise subprocess.TimeoutExpired('condor_status', 60)
        values = {'COLLECTOR_HOST': fixture_pool, 'FULL_HOSTNAME': 'local.fixture.invalid',
                  'SCHEDD_NAME': fixture_schedd}
        return response(values[command[1]] + '\n')
    monkeypatch.setattr(condor.subprocess, 'run', fake)
    with pytest.raises(backend.ChildDefinitelyNotStarted):
        condor.submit(*execution)
    assert not (execution[0].parent / 'submission_intent.json').exists()
