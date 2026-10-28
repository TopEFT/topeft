"""Per-execution HTCondor mechanics on a shared filesystem.

No plan, campaign state, retry, scientific selection or canonical output policy.
Only the semantic controller decides whether terminal evidence is successful.
"""
import json
from pathlib import Path
import re
import subprocess

from analysis.topeft_run2 import plot_campaign_backend as backend


def _quoted(value):
    value = str(value)
    if any(c in value for c in '\n\r\0'):
        raise ValueError('Unsafe Condor value')
    return '"' + value.replace('"', '""') + '"'


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


def submit(descriptor_path, descriptor):
    """Persist intent before one submit. Any post-spawn ambiguity stays unresolved."""
    metadata = descriptor_path.parent
    submit_path = metadata / 'execution.sub'
    intent_path = metadata / 'submission_intent.json'
    identity_path = metadata / 'scheduler_identity.json'
    try:
        with submit_path.open('x', encoding='utf-8') as stream:
            stream.write(submit_description(descriptor_path, descriptor))
        # Exclusive intent is the durable no-resubmit boundary.
        backend.exclusive_json(intent_path, {'token': descriptor['token']})
    except Exception as error:
        # An existing intent may correspond to an earlier successful submission.
        if intent_path.exists():
            raise backend.ActiveChildUnresolved('submission intent already exists or is incomplete') from error
        raise backend.ChildDefinitelyNotStarted(str(error)) from error
    try:
        result = subprocess.run(['condor_submit', '-terse', str(submit_path)],
                                capture_output=True, text=True, check=False, timeout=60)
    except OSError as error:
        raise backend.ChildDefinitelyNotStarted('condor_submit process could not start') from error
    except subprocess.SubprocessError as error:
        raise backend.ActiveChildUnresolved('submission outcome unknown') from error
    backend.atomic_json(metadata / 'submission_response.json', {
        'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
    })
    match = re.fullmatch(r'([0-9]+)\.0(?:\s*-\s*\1\.0)?\s*', result.stdout)
    if result.returncode != 0 or match is None:
        raise backend.ActiveChildUnresolved('submission response cannot prove a unique execution')
    backend.atomic_json(identity_path, {'job': match[1] + '.0', 'token': descriptor['token']})
    return {'state': 'pending', 'job': match[1] + '.0'}


def _query(job, command, token):
    attributes = 'ClusterId,ProcId,JobStatus,ExitCode,ExitBySignal,PlotExecutionToken,NumJobStarts'
    result = subprocess.run([command, job, '-json', '-attributes', attributes],
                            capture_output=True, text=True, check=False, timeout=60)
    if result.returncode:
        raise backend.ActiveChildUnresolved(f'{command} failed for {job}')
    ads = json.loads(result.stdout)
    if not isinstance(ads, list) or len(ads) > 1:
        raise backend.ActiveChildUnresolved('ambiguous scheduler response')
    if not ads:
        return None
    ad = ads[0]
    if (not isinstance(ad, dict) or f"{ad.get('ClusterId')}.{ad.get('ProcId')}" != job
            or ad.get('PlotExecutionToken') != token):
        raise backend.ActiveChildUnresolved('scheduler identity/token contradiction')
    return ad


def observe(descriptor_path, descriptor):
    """Exact job+token only; unknown identity never triggers submit or local fallback."""
    identity_path = descriptor_path.parent / 'scheduler_identity.json'
    if not identity_path.is_file() or identity_path.is_symlink():
        return {'state': 'unresolved', 'reason': 'no proven scheduler identity; no resubmit'}
    binding = json.loads(identity_path.read_text())
    if (binding.get('token') != descriptor['token'] or
            not re.fullmatch(r'[0-9]+\.0', binding.get('job', ''))):
        raise backend.ActiveChildUnresolved('scheduler identity binding contradiction')
    job = binding['job']
    ad = _query(job, 'condor_q', descriptor['token'])
    if ad is None:
        ad = _query(job, 'condor_history', descriptor['token'])
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
