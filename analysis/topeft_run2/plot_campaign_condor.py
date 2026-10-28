"""Per-execution HTCondor mechanics on a shared filesystem.

No plan, campaign state, retry, scientific selection or canonical output policy.
Only the semantic controller decides whether terminal evidence is successful.
"""
import json
import os
from pathlib import Path
import pwd
import re
import subprocess

from analysis.topeft_run2 import plot_campaign_backend as backend


def _quoted(value):
    value = str(value)
    if any(c in value for c in '\n\r\0'):
        raise ValueError('Unsafe Condor value')
    return '"' + value.replace('"', '""') + '"'


def _run(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        raise backend.ActiveChildUnresolved(f'{command[0]} unavailable') from error


def _config(name, optional=False):
    result = _run(['condor_config_val', name])
    if result.returncode:
        if optional and result.returncode == 1 and result.stderr.strip().startswith('Not defined:'):
            return None
        raise backend.ChildDefinitelyNotStarted(f'Condor {name} configuration unavailable')
    value = result.stdout.strip()
    if (not value and optional):
        return None
    if not value or '\n' in value or '\r' in value:
        raise backend.ChildDefinitelyNotStarted(f'Condor {name} configuration ambiguous')
    return value


def _target_value(value, label):
    if (not isinstance(value, str) or not value or value.startswith('-') or
            re.search(r'[\s,;\x00-\x1f]', value)):
        raise backend.ActiveChildUnresolved(f'invalid persisted {label}')
    return value


def resolve_target():
    """Resolve one local schedd from installed configuration and collector ads."""
    try:
        pool = _config('COLLECTOR_HOST')
        host = _config('FULL_HOSTNAME')
        configured = _config('SCHEDD_NAME', optional=True)
    except backend.ActiveChildUnresolved as error:
        raise backend.ChildDefinitelyNotStarted('Condor configuration lookup unavailable') from error
    try:
        _target_value(pool, 'pool')
    except backend.ActiveChildUnresolved as error:
        raise backend.ChildDefinitelyNotStarted('Condor pool is not one explicit target') from error
    candidate = configured or host
    constraint = 'Name == ' + json.dumps(candidate) if configured else 'Machine == ' + json.dumps(host)
    try:
        result = _run(['condor_status', '-pool', pool, '-schedd', '-constraint', constraint,
                       '-json', '-attributes', 'Name,Machine'])
    except backend.ActiveChildUnresolved as error:
        raise backend.ChildDefinitelyNotStarted('local Condor schedd lookup unavailable') from error
    if result.returncode:
        raise backend.ChildDefinitelyNotStarted('local Condor schedd lookup failed')
    try:
        ads = json.loads(result.stdout)
    except ValueError as error:
        raise backend.ChildDefinitelyNotStarted('local Condor schedd ads malformed') from error
    if (not isinstance(ads, list) or len(ads) != 1 or not isinstance(ads[0], dict)
            or ads[0].get('Machine') != host or not isinstance(ads[0].get('Name'), str)
            or (configured and ads[0]['Name'] != configured)):
        raise backend.ChildDefinitelyNotStarted('local Condor schedd identity is not unique')
    schedd = ads[0]['Name']
    try:
        _target_value(schedd, 'schedd')
    except backend.ActiveChildUnresolved as error:
        raise backend.ChildDefinitelyNotStarted('local Condor schedd name is unsafe') from error
    return {'pool': pool, 'schedd': schedd}


def submit_description(descriptor_path, descriptor):
    """Exactly one worker; automatic job restarts are held after first launch."""
    metadata = Path(descriptor_path).parent
    argv = ['-m', 'analysis.topeft_run2.plot_campaign_backend', '--worker', str(descriptor_path)]
    # HTCondor new argument syntax: each argument uses single quotes inside an
    # outer double-quoted string. Double both quote kinds inside arguments.
    arguments = '"' + ' '.join("'" + a.replace('"', '""').replace("'", "''") + "'" for a in argv) + '"'
    return '\n'.join([
        'universe = vanilla',
        'executable = ' + _quoted(descriptor['argv'][0]),
        'arguments = ' + arguments,
        'initialdir = ' + _quoted(backend.REPO_ROOT),
        'environment = ' + _quoted('PYTHONPATH=' + str(backend.REPO_ROOT)),
        'getenv = False',
        'should_transfer_files = NO',
        'request_cpus = 1',
        f"request_memory = {(descriptor['hard_limit_bytes'] + 1048575) // 1048576}",
        'request_disk = 10240',
        '+MaxRuntime = 86400',
        '+PlotExecutionToken = ' + _quoted(descriptor['token']),
        'on_exit_hold = (ExitBySignal == True) || (ExitCode != 0)',
        'periodic_hold = (NumJobStarts > 1)',
        'output = ' + _quoted(metadata / 'worker.out'),
        'error = ' + _quoted(metadata / 'worker.err'),
        'log = ' + _quoted(metadata / 'scheduler.log'),
        'queue 1', '',
    ])


def _read_intent(path, token):
    if not path.is_file() or path.is_symlink():
        raise backend.ActiveChildUnresolved('submission intent unavailable')
    try:
        intent = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as error:
        raise backend.ActiveChildUnresolved('submission intent malformed') from error
    if not isinstance(intent, dict) or intent.get('token') != token:
        raise backend.ActiveChildUnresolved('submission intent token contradiction')
    _target_value(intent.get('pool'), 'pool')
    _target_value(intent.get('schedd'), 'schedd')
    return intent


def _read_identity(path, intent):
    if not path.is_file() or path.is_symlink():
        raise backend.ActiveChildUnresolved('scheduler identity unavailable')
    try:
        binding = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, ValueError) as error:
        raise backend.ActiveChildUnresolved('scheduler identity malformed') from error
    if (not isinstance(binding, dict) or
            any(binding.get(key) != intent[key] for key in ('token', 'pool', 'schedd')) or
            not isinstance(binding.get('job'), str) or
            not re.fullmatch(r'[0-9]+\.0', binding['job'])):
        raise backend.ActiveChildUnresolved('scheduler identity binding contradiction')
    return binding


def _persist_identity(path, intent, job):
    binding = {**{key: intent[key] for key in ('token', 'pool', 'schedd')}, 'job': job}
    try:
        backend.exclusive_json(path, binding)
    except FileExistsError:
        if _read_identity(path, intent) != binding:
            raise backend.ActiveChildUnresolved('concurrent scheduler identity contradiction')
    except OSError as error:
        raise backend.ActiveChildUnresolved('scheduler identity durability unavailable') from error
    return binding


def submit(descriptor_path, descriptor):
    """Persist target and intent before one submit; later ambiguity never retries."""
    metadata = descriptor_path.parent
    submit_path = metadata / 'execution.sub'
    intent_path = metadata / 'submission_intent.json'
    identity_path = metadata / 'scheduler_identity.json'
    if (intent_path.exists() or intent_path.is_symlink() or
            identity_path.exists() or identity_path.is_symlink()):
        raise backend.ActiveChildUnresolved('scheduler metadata already exists')
    target = resolve_target()
    try:
        with submit_path.open('x', encoding='utf-8') as stream:
            stream.write(submit_description(descriptor_path, descriptor))
        backend.exclusive_json(intent_path, {'token': descriptor['token'], **target})
    except Exception as error:
        if (intent_path.exists() or intent_path.is_symlink() or
                submit_path.exists() or submit_path.is_symlink()):
            raise backend.ActiveChildUnresolved('submission setup occupied or incomplete') from error
        raise backend.ChildDefinitelyNotStarted(str(error)) from error
    intent = _read_intent(intent_path, descriptor['token'])
    try:
        result = _run(['condor_submit', '-pool', intent['pool'], '-name', intent['schedd'],
                       '-terse', str(submit_path)])
    except backend.ActiveChildUnresolved as error:
        raise backend.ActiveChildUnresolved('submission outcome unknown') from error
    try:
        backend.atomic_json(metadata / 'submission_response.json', {
            'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
        })
    except OSError as error:
        raise backend.ActiveChildUnresolved('submission response durability unavailable') from error
    match = re.fullmatch(r'([0-9]+)\.0(?:\s*-\s*\1\.0)?\s*', result.stdout)
    if result.returncode != 0 or match is None:
        raise backend.ActiveChildUnresolved('submission response cannot prove a unique execution')
    job = match[1] + '.0'
    _persist_identity(identity_path, intent, job)
    return {'state': 'pending', 'job': job}


def _query(job, command, intent):
    attributes = 'ClusterId,ProcId,Owner,JobStatus,ExitCode,ExitBySignal,PlotExecutionToken,NumJobStarts'
    result = _run([command, '-pool', intent['pool'], '-name', intent['schedd'], job,
                   '-json', '-attributes', attributes])
    if result.returncode:
        raise backend.ActiveChildUnresolved(f'{command} failed for {job}')
    try:
        # HTCondor may emit no JSON bytes for a successful zero-ad query.
        ads = json.loads(result.stdout) if result.stdout.strip() else []
    except ValueError as error:
        raise backend.ActiveChildUnresolved('malformed scheduler response') from error
    if not isinstance(ads, list) or len(ads) > 1:
        raise backend.ActiveChildUnresolved('ambiguous scheduler response')
    if not ads:
        return None
    ad = ads[0]
    if (not isinstance(ad, dict) or type(ad.get('ClusterId')) is not int or
            type(ad.get('ProcId')) is not int or ad['ProcId'] != 0 or
            f"{ad['ClusterId']}.0" != job or
            ad.get('PlotExecutionToken') != intent['token'] or
            ad.get('Owner') != pwd.getpwuid(os.getuid()).pw_name):
        raise backend.ActiveChildUnresolved('scheduler identity/token/owner contradiction')
    return ad


def _recover(command, intent):
    owner = pwd.getpwuid(os.getuid()).pw_name
    constraint = ('PlotExecutionToken == ' + json.dumps(intent['token']) +
                  ' && Owner == ' + json.dumps(owner))
    attributes = 'ClusterId,ProcId,Owner,PlotExecutionToken,JobStatus,ExitCode,ExitBySignal,NumJobStarts'
    result = _run([command, '-pool', intent['pool'], '-name', intent['schedd'],
                   '-constraint', constraint, '-limit', '2', '-json', '-attributes', attributes])
    if result.returncode:
        raise backend.ActiveChildUnresolved(f'{command} recovery query failed')
    try:
        ads = json.loads(result.stdout) if result.stdout.strip() else []
    except ValueError as error:
        raise backend.ActiveChildUnresolved('malformed recovery response') from error
    if not isinstance(ads, list) or len(ads) > 1:
        raise backend.ActiveChildUnresolved('ambiguous recovery response')
    if not ads:
        return None
    ad = ads[0]
    if (not isinstance(ad, dict) or type(ad.get('ClusterId')) is not int or
            ad['ClusterId'] < 0 or type(ad.get('ProcId')) is not int or ad['ProcId'] != 0 or
            ad.get('Owner') != owner or ad.get('PlotExecutionToken') != intent['token']):
        raise backend.ActiveChildUnresolved('recovery identity/token/owner contradiction')
    return f"{ad['ClusterId']}.0"


def observe(descriptor_path, descriptor):
    """Observe only the persisted target; missing identity reconciles without submit."""
    intent_path = descriptor_path.parent / 'submission_intent.json'
    identity_path = descriptor_path.parent / 'scheduler_identity.json'
    if not intent_path.exists() and not intent_path.is_symlink():
        if identity_path.exists() or identity_path.is_symlink():
            raise backend.ActiveChildUnresolved('scheduler identity without intent')
        return {'state': 'unresolved', 'reason': 'no proven submission intent; no resubmit'}
    intent = _read_intent(intent_path, descriptor['token'])
    if identity_path.exists() or identity_path.is_symlink():
        job = _read_identity(identity_path, intent)['job']
    else:
        job = _recover('condor_q', intent)
        if job is None:
            job = _recover('condor_history', intent)
        if job is None:
            return {'state': 'unresolved', 'reason': 'no unique scheduler identity; no resubmit'}
        _persist_identity(identity_path, intent, job)
    ad = _query(job, 'condor_q', intent)
    if ad is None:
        ad = _query(job, 'condor_history', intent)
    if ad is None:
        return {'state': 'unresolved', 'job': job}
    status = ad.get('JobStatus')
    # Held jobs are not terminal: they can be released externally.
    if status == 3:
        state = 'terminal_failure'
    elif status == 4:
        if (type(ad.get('ExitCode')) is not int or type(ad.get('ExitBySignal')) is not bool
                or type(ad.get('NumJobStarts')) is not int):
            state = 'unresolved'
        else:
            state = ('terminal_success' if ad['ExitCode'] == 0 and
                     not ad['ExitBySignal'] and ad['NumJobStarts'] == 1 else 'terminal_failure')
    else:
        state = 'pending' if status in {1, 2, 5, 6, 7} else 'unresolved'
    return {'state': state, 'job': job, 'ad': ad}
