#!/usr/bin/env python3
"""Read-only Work Queue / Condor dashboard. Python 3.9+, standard library only."""
import collections
import concurrent.futures
import datetime
import getpass
import json
import math
import os
import pwd
import re
import shlex
import signal
import stat
import statistics
import subprocess
import sys
import time

VERSION = '2026-10-26.1'


def normalized_summary(record):
    """The catalog's lean CCTools record uses tasks_complete, not tasks_done."""
    row = dict(record)
    for key, alias in (('tasks_done', 'tasks_complete'), ('workers_connected', 'workers')):
        if number(row.get(key)) is None and number(row.get(alias)) is not None:
            row[key] = row[alias]
    return row


def matching_catalog(records, host, port, owner):
    return [normalized_summary(r) for r in records or []
            if str(r.get('port')) == str(port)
            and r.get('owner') == owner
            and host.rstrip('.').lower() in
            [str(r.get(k, '')).rstrip('.').lower() for k in ('name', 'hostname', 'address')]]


def owner_catalog(records, owner, now, max_age):
    """Only explicit owner records; freshness is a property of each snapshot."""
    result = []
    for record in records or []:
        if record.get('owner') != owner:
            continue
        row = normalized_summary(record)
        heard = number(row.get('lastheardfrom'))
        age = now - heard if heard is not None else None
        row['_age'] = age
        row['_fresh'] = age is not None and -5 <= age <= max_age
        result.append(row)
    return result


def select_project(rows, project):
    fresh = [r for r in rows if r['_fresh'] and r.get('project') == project]
    matches = [r for r in fresh if (r.get('name') or r.get('hostname') or r.get('address'))
               and number(r.get('port')) is not None]
    if len(matches) != 1:
        if matches:
            return None, 'AMBIGUOUS_FRESH_MATCH'
        if fresh:
            return None, 'FRESH_RECORD_WITHOUT_ENDPOINT'
        if any(r.get('project') == project and not r['_fresh'] for r in rows):
            return None, 'STALE_PROJECT_ONLY'
        return None, 'NO_FRESH_MATCH'
    selected = matches[0]
    return (str(selected.get('name') or selected.get('hostname') or selected['address']),
            str(selected['port'])), 'OK'


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) and n >= 0 else None
    except (TypeError, ValueError):
        return None


def fmt(value):
    n = number(value)
    return '?' if n is None else ('%g' % n)


def clean(value):
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', str(value))


def table(headers, rows):
    rows = [[clean(x) for x in row] for row in rows]
    if not rows:
        print('<none>')
        return
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    for row in [headers] + rows:
        print('  '.join(x.ljust(w) for x, w in zip(row, widths)))


def section(name):
    print('\n=== %s ===' % name)


def run(args, timeout, input_text=None):
    try:
        p = subprocess.Popen(args, stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             encoding='utf-8', errors='replace', start_new_session=True)
    except OSError as e:
        return 127, '', str(e)
    try:
        out, err = p.communicate(input=input_text, timeout=timeout)
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.communicate()
        return 124, '', 'query deadline exceeded'


def decode_records(text):
    # CCTools long output may be an array or successive JSON objects.
    decoder = json.JSONDecoder()
    remaining = text.strip()
    if not remaining:
        raise ValueError('empty response; not a confirmed empty queue')
    rows = []
    while remaining:
        value, end = decoder.raw_decode(remaining)
        batch = value if isinstance(value, list) else [value]
        if not all(isinstance(x, dict) for x in batch):
            raise ValueError('expected JSON objects')
        rows.extend(batch)
        remaining = remaining[end:].lstrip()
    return rows


def query(args, timeout):
    rc, out, err = run(args, timeout)
    if rc:
        return None, 'TIMEOUT' if rc == 124 else 'FAILED rc=%d %s' % (rc, clean(err or out)[:200])
    try:
        rows = decode_records(out)
        return rows, 'OK'
    except ValueError as e:
        return None, 'INVALID_RESPONSE: ' + str(e)


def manager_queries(host, port, timeout, task_details, task_timeout, query_fn=None,
                    expected_project=None):
    query_fn = query_fn or query
    if host is None or port is None:
        return None, 'NO_SELECTED_ENDPOINT', None, 'DISABLED' if not task_details else 'NO_SELECTED_ENDPOINT'
    base = ['work_queue_status', '-l', '-t', str(math.ceil(timeout))]
    summary, summary_status = query_fn(base + [host, port], timeout)
    if expected_project is not None:
        if summary_status != 'OK' or summary is None or len(summary) != 1 or not isinstance(summary[0], dict):
            return None, summary_status, None, ('DISABLED' if not task_details else 'SKIPPED_UNVERIFIED_DIRECT_PROJECT')
        if summary[0].get('project') != expected_project:
            return None, 'DIRECT_PROJECT_MISMATCH', None, ('DISABLED' if not task_details else 'SKIPPED_DIRECT_PROJECT_MISMATCH')
    if not task_details:
        return summary, summary_status, None, 'DISABLED'
    task_base = ['work_queue_status', '-l', '-t', str(math.ceil(task_timeout)), '-T', host, port]
    tasks, task_status = query_fn(task_base, task_timeout)
    return summary, summary_status, tasks, task_status


def worker_origin(job):
    cmd = job.get('Cmd', '')
    basename = os.path.basename(str(cmd))
    if basename == 'work_queue_worker':
        return os.path.dirname(str(cmd)) or str(cmd)
    if basename not in ('condor.sh', 'with_oasis_certs'):
        return None
    args = job.get('Args') or job.get('Arguments') or ''
    try:
        argv = shlex.split(str(args))
    except ValueError:
        return None
    wrapped_worker = (bool(argv) and
                      (os.path.basename(argv[0]) == 'work_queue_worker' or
                       (basename == 'condor.sh' and os.path.basename(argv[0]) == 'with_oasis_certs'
                        and len(argv) > 1 and os.path.basename(argv[1]) == 'work_queue_worker')))
    if not wrapped_worker:
        return None
    # Provenance only; it does not imply a task category or manager assignment.
    return os.path.dirname(str(cmd)) or str(cmd)


def option_value(argv, *names):
    for index, arg in enumerate(argv):
        if arg in names and index + 1 < len(argv):
            return argv[index + 1]
        for name in names:
            if arg.startswith(name + '='):
                return arg[len(name) + 1:]
    return None


def worker_selector(job):
    try:
        args = shlex.split(str(job.get('Args') or job.get('Arguments') or ''))
    except ValueError:
        return '?'
    return option_value(args, '-M', '--manager-name') or '?'


def resource_profile(job):
    disk = number(job.get('RequestDisk'))
    return (fmt(job.get('RequestCpus')), fmt(job.get('RequestMemory')),
            fmt(disk / 1024 if disk is not None else None))


def pool_key(job):
    return (worker_origin(job), worker_selector(job)) + resource_profile(job)


def pool_counts(jobs):
    pools = collections.defaultdict(collections.Counter)
    for job in jobs:
        if worker_origin(job) is not None:
            pools[pool_key(job)][job.get('JobStatus')] += 1
    return pools


def held_job_attribution(job, factory_scratch):
    origin = worker_origin(job)
    if origin is None:
        return '?', '?', '?'
    binding = 'LIVE_FACTORY_BOUND' if origin in factory_scratch else 'NO_LIVE_FACTORY_MATCH'
    return origin, binding, worker_selector(job)


def factory_from_process(line):
    """Inspect one pgrep PID/argv line; no process or config is changed."""
    match = re.match(r'^\s*(\d+)\s+(.*)$', line)
    if not match:
        return None
    try:
        argv = shlex.split(match.group(2))
    except ValueError:
        return None
    if not argv or os.path.basename(argv[0]) != 'work_queue_factory':
        return None
    config_path = option_value(argv, '-C', '--config-file')
    config, config_status = {}, 'NOT_REFERENCED'
    if config_path:
        try:
            read_path = config_path
            if not os.path.isabs(read_path):
                read_path = os.path.join(os.readlink('/proc/%s/cwd' % match.group(1)), read_path)
            config_stat = os.stat(read_path)
            if not stat.S_ISREG(config_stat.st_mode) or config_stat.st_size > 1048576:
                raise ValueError('not a regular config <= 1 MiB')
            with open(read_path, encoding='utf-8') as stream:
                config = json.load(stream)
            if not isinstance(config, dict):
                raise ValueError('config is not an object')
            config_status = 'OK'
        except (OSError, UnicodeError, ValueError) as exc:
            config = {}
            config_status = 'UNAVAILABLE: ' + clean(exc)[:120]
    option_names = {'min-workers': ('-w', '--min-workers'),
                    'max-workers': ('-W', '--max-workers'),
                    'workers-per-cycle': ('--workers-per-cycle',),
                    'cores': ('--cores',), 'memory': ('--memory',), 'disk': ('--disk',)}
    settings = {key: option_value(argv, *names) or config.get(key)
                for key, names in option_names.items()}
    settings['worker_timeout'] = config.get('timeout')
    settings['factory_timeout'] = option_value(argv, '--factory-timeout')
    return {'pid': match.group(1), 'argv': argv, 'config_path': config_path,
            'config_status': config_status, 'selector': option_value(argv, '-M', '-N', '--manager-name') or config.get('manager-name') or '?',
            'scratch': option_value(argv, '--scratch-dir', '-S') or config.get('scratch-dir'),
            'config': config, 'settings': settings}


def factory_warnings(factory, jobs):
    scratch = factory['scratch']
    active = [j for j in jobs if scratch and worker_origin(j) == scratch and j.get('JobStatus') != 4]
    counts = collections.Counter(j.get('JobStatus') for j in active)
    max_workers = number(factory['settings'].get('max-workers'))
    warning = ('OBSERVED_ACTIVE_JOBS_ABOVE_DECLARED_MAX' if max_workers is not None
               and len(active) > max_workers else None)
    return counts, warning


CGROUP_READ_SCRIPT = r'''
set -f
for pid in ${_CONDOR_JOB_PIDS-}; do
  case "$pid" in ''|*[!0-9]*) continue ;; esac
  while IFS=: read -r hierarchy controllers group; do
    if [[ "$hierarchy" == 0 && -z "$controllers" && "$group" == /* ]]; then
      base="/sys/fs/cgroup${group}"
      for name in memory.current memory.peak memory.max memory.events; do
        if [[ -r "$base/$name" ]]; then
          printf '%s: ' "$name"
          cat "$base/$name"
        else
          printf '%s: UNAVAILABLE\n' "$name"
        fi
      done
      exit 0
    fi
  done < "/proc/$pid/cgroup"
  exit 1
done
exit 1
'''


def cgroup_probes(requested_ids, jobs, owner, schedd, max_probes, timeout, run_fn=None):
    """Probe only explicitly selected RUNNING jobs from this owner's snapshot."""
    if not requested_ids or jobs is None or max_probes <= 0:
        return []
    run_fn = run_fn or run
    eligible = {'%s.%s' % (j.get('ClusterId'), j.get('ProcId')) for j in jobs
                if j.get('Owner') == owner and j.get('JobStatus') == 2
                and isinstance(j.get('ClusterId'), int) and isinstance(j.get('ProcId'), int)
                and not isinstance(j.get('ClusterId'), bool) and not isinstance(j.get('ProcId'), bool)
                and j['ClusterId'] >= 0 and j['ProcId'] >= 0}
    results = []
    seen = set()
    for job_id in re.split(r'[,\s]+', requested_ids.strip()):
        if not re.fullmatch(r'\d+\.\d+', job_id) or job_id not in eligible or job_id in seen:
            continue
        seen.add(job_id)
        rc, out, err = run_fn(['condor_ssh_to_job', '-name', schedd, job_id, 'bash', '-s'],
                              timeout, input_text=CGROUP_READ_SCRIPT)
        results.append((job_id, rc, out[:4096], clean(err)[:200]))
        if len(results) >= max_probes:
            break
    return results


def recent_exits(records):
    grouped = collections.defaultdict(list)
    for job in records or []:
        origin = worker_origin(job)
        if origin is not None:
            grouped[origin].append(job)
    result = {}
    for origin, jobs in grouped.items():
        codes = collections.Counter()
        runtimes = []
        exit_zero = exit_nonzero = signaled = 0
        for job in jobs:
            code = job.get('ExitCode')
            by_signal = job.get('ExitBySignal') is True or number(job.get('ExitSignal')) not in (None, 0)
            if by_signal:
                signaled += 1
            elif isinstance(code, int) and not isinstance(code, bool):
                codes[code] += 1
                if code == 0:
                    exit_zero += 1
                else:
                    exit_nonzero += 1
            runtime = number(job.get('RemoteWallClockTime'))
            if runtime is not None:
                runtimes.append(runtime)
        result[origin] = (len(jobs), exit_zero, exit_nonzero, signaled,
                          statistics.median(runtimes) if runtimes else None, codes)
    return result


def manager_alerts(rows):
    return [r for r in rows if r['_fresh'] and
            (number(r.get('tasks_waiting')) or 0) > 0 and
            number(r.get('workers_connected')) == 0]


def supply_alerts(jobs, history, factories):
    alerts = []
    for key, counts in sorted(pool_counts(jobs).items()):
        if counts[5]:
            alerts.append(('HELD_WORKER_JOBS', key[0], key[1], counts[5]))
    for origin, row in sorted(recent_exits(history).items()):
        if row[2] or row[3]:
            alerts.append(('RECENT_NONZERO_OR_SIGNALED', origin, '?', row[2] + row[3]))
    for factory in factories:
        _, warning = factory_warnings(factory, jobs)
        if warning:
            alerts.append((warning, factory['scratch'], factory['selector'], factory['pid']))
    return alerts


def stats_tail(path):
    with open(path, 'rb') as f:
        header = f.readline().decode().lstrip('# ').split()
        start = f.tell()
        end = f.seek(0, 2)
        offset = max(start, end - 131072)
        f.seek(offset)
        if offset > start:
            f.readline()  # discard potentially partial first record
        raw = f.read()
    lines = raw.splitlines()
    if raw and not raw.endswith(b'\n'):
        lines = lines[:-1]
    rows = []
    for line in lines:
        values = line.decode().split()
        if len(values) == len(header):
            row = dict(zip(header, values))
            if number(row.get('timestamp')) is not None:
                rows.append(row)
    if not rows:
        raise ValueError('no complete stats samples')
    last = rows[-1]
    stamp = float(last['timestamp']) / 1e6
    earlier = next((r for r in reversed(rows[:-1]) if stamp - float(r['timestamp']) / 1e6 >= 30), None)
    if earlier is None and len(rows) > 1:
        earlier = rows[0]
    return last, earlier, stamp


def main():
    if '--help' in sys.argv:
        print('''Usage: ./wq_dashboard.py   (python3; no pip packages)
Read-only, no worker/job removals or configuration changes.
Environment:
  MANAGER_PROJECT=exact-name      resolve one fresh owner catalog endpoint
  MANAGER_HOST=glados.crc.nd.edu MANAGER_PORT=9164
  OWNER=current-user SCHEDD=condorfe.crc.nd.edu
  QUERY_TIMEOUT=10 TASK_DETAILS=0 TASK_QUERY_TIMEOUT=10
  MAX_ACTIVE_TASKS=8            display limit only; not an RPC cost limit
  HISTORY_LOOKBACK_SECONDS=300 HISTORY_MAX_JOBS=500
  HISTORY_QUERY_TIMEOUT=20        independent bounded history deadline
  CGROUP_JOB_IDS=123.0,124.0    explicit RUNNING owner-snapshot jobs only
  MAX_CGROUP_PROBES=4 CGROUP_PROBE_TIMEOUT=8  bounded opt-in SSH reads
  LOGDIR=/path/to/source_native  optional shared, readable manager logs
  STATS_MAX_AGE=120              older stats never establish current health
Prefer watch -n 10 python3 wq_dashboard.py. Task enumeration is opt-in and
queries only the selected manager; it may return a large payload.
Updates can take longer than the watch interval. Catalog and direct queries
are separate snapshots. Condor profiles cover this owner's identified WQ
jobs across all managers; they are not manager-specific capacity.
Catalog records lacking explicit owner are omitted. Stale records are labeled.
Condor RUNNING does not prove Work Queue connection or task activity.
Matching task hostname to Condor is advisory only, never proof of identity.
Host/task matching alone does not authorize a cgroup probe.
For running tasks, resources shown are allocated, not measured utilization.
LOGDIR must refer to this manager/run; no automatic log discovery is done.
No completion or monitor-deadlock diagnosis is inferred from inactivity.''')
        return
    host = os.environ.get('MANAGER_HOST', 'glados.crc.nd.edu')
    port = os.environ.get('MANAGER_PORT', '9164')
    project = os.environ.get('MANAGER_PROJECT')
    owner = os.environ.get('OWNER') or getpass.getuser()
    schedd = os.environ.get('SCHEDD', 'condorfe.crc.nd.edu')
    limit = max(1, int(os.environ.get('MAX_ACTIVE_TASKS', '8')))
    timeout = max(1., float(os.environ.get('QUERY_TIMEOUT', '10')))
    task_timeout = max(1., float(os.environ.get('TASK_QUERY_TIMEOUT', '10')))
    history_timeout = min(60., max(1., float(os.environ.get('HISTORY_QUERY_TIMEOUT', '20'))))
    task_setting = os.environ.get('TASK_DETAILS', '0')
    if task_setting not in ('0', '1'):
        raise ValueError('TASK_DETAILS must be 0 or 1')
    task_details = task_setting == '1'
    lookback = min(86400, max(1, int(os.environ.get('HISTORY_LOOKBACK_SECONDS', '300'))))
    history_max = min(5000, max(1, int(os.environ.get('HISTORY_MAX_JOBS', '500'))))
    age_limit = max(1., float(os.environ.get('STATS_MAX_AGE', '120')))
    cgroup_ids = os.environ.get('CGROUP_JOB_IDS', '')
    max_cgroup_probes = min(16, max(0, int(os.environ.get('MAX_CGROUP_PROBES', '4'))))
    cgroup_probe_timeout = min(30., max(1., float(os.environ.get('CGROUP_PROBE_TIMEOUT', '8'))))
    now = time.time()
    print('timestamp:', datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
    print('dashboard:', VERSION)
    print('manager_selector: %s  owner: %s  schedd: %s' %
          (clean('project=' + project if project else 'endpoint=%s:%s' % (host, port)), clean(owner), clean(schedd)))
    attrs = ('ClusterId,ProcId,Owner,JobStatus,RemoteHost,LastRemoteHost,RequestCpus,RequestMemory,'
             'RequestDisk,MemoryUsage,DiskUsage,Cmd,Args,Arguments,HoldReason,HoldReasonCode,QDate')
    history_attrs = ('ClusterId,ProcId,Owner,JobStatus,Cmd,Args,Arguments,CompletionDate,ExitCode,'
                     'ExitBySignal,ExitSignal,RemoteWallClockTime')
    history_constraint = ('Owner == %s && CompletionDate >= %d' % (json.dumps(owner), int(now - lookback)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        catalog_f = pool.submit(query, ['work_queue_status', '-l', '-t', str(math.ceil(timeout))], timeout)
        condor_f = pool.submit(query, ['condor_q', '-name', schedd, '-constraint', 'Owner == ' + json.dumps(owner),
                                       '-json', '-attributes', attrs], timeout)
        history_f = pool.submit(query, ['condor_history', '-name', schedd, '-constraint', history_constraint,
                                        '-scanlimit', str(history_max * 10), '-limit', str(history_max),
                                        '-json', '-attributes', history_attrs], history_timeout)
        catalog, catalog_status = catalog_f.result()
        jobs, condor_status = condor_f.result()
        history, history_status = history_f.result()
    history_raw_count = len(history) if history is not None else None
    if jobs is not None:
        jobs = [j for j in jobs if j.get('Owner') == owner]
    if history is not None:
        history = [j for j in history if j.get('Owner') == owner and
                   number(j.get('CompletionDate')) is not None and
                   number(j.get('CompletionDate')) >= now - lookback]
    owner_managers = owner_catalog(catalog, owner, time.time(), age_limit)
    selection_status = 'LEGACY_ENDPOINT'
    if project is not None:
        endpoint, selection_status = select_project(owner_managers, project)
        host, port = endpoint if endpoint else (None, None)
    queues, queue_status, tasks, task_status = manager_queries(
        host, port, timeout, task_details, task_timeout, expected_project=project)
    if queue_status == 'DIRECT_PROJECT_MISMATCH':
        selection_status = queue_status
    queue = normalized_summary(queues[0]) if queues and len(queues) == 1 else None
    if queue is not None and (number(queue.get('tasks_waiting')) is None or number(queue.get('tasks_running')) is None):
        queue = None
    if queue is None and queue_status == 'OK':
        queue_status = 'INVALID_RESPONSE: expected one manager summary'
    if tasks is not None and any(number(t.get('taskid')) is None or not isinstance(t.get('state'), str) for t in tasks):
        tasks, task_status = None, 'INVALID_RESPONSE: task schema'

    if project is not None:
        selected = ([r for r in owner_managers if r['_fresh'] and r.get('project') == project
                     and str(r.get('port')) == port
                     and str(r.get('name') or r.get('hostname') or r.get('address')) == host]
                    if selection_status == 'OK' else [])
    else:
        selected = matching_catalog(catalog, host, port, owner)
    catalog_queue = selected[0] if len(selected) == 1 else None
    identity = queue or catalog_queue
    logdir = os.environ.get('LOGDIR')
    log_sample, log_error, log_fresh = None, None, False
    if logdir:
        try:
            log_sample = stats_tail(os.path.join(logdir, 'stats.log'))
            last, first, stamp = log_sample
            log_age = time.time() - stamp
            log_fresh = -5 <= log_age <= age_limit
            if identity and number(identity.get('starttime')) is not None and stamp < float(identity['starttime']):
                log_fresh = False
                log_error = 'stats predates current manager'
        except (OSError, ValueError, KeyError) as e:
            log_error = clean(e)
    catalog_age = (time.time() - float(catalog_queue['lastheardfrom'])
                   if catalog_queue and number(catalog_queue.get('lastheardfrom')) is not None else None)
    catalog_fresh = catalog_age is not None and -5 <= catalog_age <= age_limit

    section('managers_catalog')
    print('query_status:', catalog_status)
    if catalog is None:
        print('UNAVAILABLE')
    else:
        print('owner_records=%d omitted_missing_or_other_owner=%d' %
              (len(owner_managers), len(catalog) - len(owner_managers)))
        table(['project', 'host', 'port', 'age', 'freshness', 'waiting', 'running', 'done', 'workers', 'workers_idle'],
              [[r.get('project', '?'), r.get('name', r.get('address', '?')), r.get('port', '?'),
                'unknown' if r['_age'] is None else '%.0fs' % r['_age'],
                'FRESH' if r['_fresh'] else 'STALE/UNKNOWN_AGE'] +
               [fmt(r.get(k)) for k in ('tasks_waiting', 'tasks_running', 'tasks_done',
                                        'workers_connected', 'workers_idle')]
               for r in owner_managers])
        print('Catalog presence is a snapshot, not direct manager liveness.')

    section('selected_manager')
    print('selection:', selection_status)
    print('endpoint:', clean('%s:%s' % (host, port) if host is not None else 'UNAVAILABLE'))
    section('work_queue')
    print('direct_query:', queue_status)
    display_queue = queue
    if queue is not None:
        print('summary_source: DIRECT')
    elif log_sample and log_fresh:
        display_queue = log_sample[0]
        print('summary_source: STATS_LOG age=%.0fs (explicit LOGDIR; direct query unavailable)' % log_age)
    elif catalog_queue is not None:
        display_queue = catalog_queue
        print('summary_source: CATALOG age=%s %s' %
              ('unknown' if catalog_age is None else '%.0fs' % catalog_age,
               'RECENT' if catalog_fresh else 'STALE/UNKNOWN_AGE'))
    if display_queue is not None:
        print('project:', clean((identity or {}).get('project', '?')))
        table(['waiting', 'running', 'on_workers', 'results', 'done', 'workers', 'workers_idle'],
              [[fmt(display_queue.get(k)) for k in ('tasks_waiting', 'tasks_running', 'tasks_on_workers',
                'tasks_with_results', 'tasks_done', 'workers_connected', 'workers_idle')]])
    print('catalog_query:', catalog_status)
    if catalog is not None:
        for r in selected:
            heard = number(r.get('lastheardfrom'))
            age = 'unknown' if heard is None else '%.0fs' % (time.time() - heard)
            print('catalog snapshot: project=%s waiting=%s running=%s done=%s age=%s' %
                  (clean(r.get('project', '?')), fmt(r.get('tasks_waiting')), fmt(r.get('tasks_running')), fmt(r.get('tasks_done')), age))
        if not selected:
            print('<no exact catalog match; not evidence that manager is absent>')

    section('work_queue_categories')
    categories = queue.get('categories') if queue else None
    if isinstance(categories, list) and all(isinstance(c, dict) for c in categories):
        table(['category', 'waiting', 'running', 'on_workers', 'fit_reported', 'max_cores', 'max_memory', 'max_disk'],
              [[clean(c.get(k, '?')) for k in ('category', 'tasks_waiting', 'tasks_running', 'tasks_on_workers',
                'workers_able', 'max_cores', 'max_memory', 'max_disk')] for c in categories])
        print('fit_reported: manager estimate, not a count of idle slots available now.')
    else:
        if task_status == 'DISABLED':
            print('DISABLED: task fallback requires TASK_DETAILS=1')
        elif tasks is not None:
            print('source: TASK_QUERY (counts only; category resource estimates unavailable)')
            groups = collections.defaultdict(collections.Counter)
            for task in tasks:
                groups[str(task.get('category', '?'))][task['state'].upper()] += 1
            table(['category', 'ready', 'running', 'waiting_retrieval', 'other'],
                  [[name, str(c['READY']), str(c['RUNNING']), str(c['WAITING_RETRIEVAL']),
                    str(sum(v for k,v in c.items() if k not in ('READY','RUNNING','WAITING_RETRIEVAL')))]
                   for name,c in sorted(groups.items())])
        else:
            print('<unavailable: neither summary categories nor task query returned>')

    section('active_work_queue_tasks')
    print('query_status:', task_status)
    active = [t for t in tasks or [] if t['state'].upper() == 'RUNNING']
    print('running_tasks:', len(active) if tasks is not None else 'DISABLED' if task_status == 'DISABLED' else 'UNKNOWN')
    if tasks is not None:
        print('states:', ', '.join('%s=%d' % x for x in sorted(collections.Counter(t['state'] for t in tasks).items())) or '<empty>')
        active.sort(key=lambda t: number(t.get('memory')) or 0, reverse=True)
        table(['task_id', 'category', 'host', 'cores', 'memory_MB', 'disk_MB', 'gpus'],
              [[str(t['taskid']), t.get('category', '?'), t.get('host', '?')] +
               [fmt(t.get(k)) for k in ('cores', 'memory', 'disk', 'gpus')] for t in active[:limit]])
        if len(active) > limit:
            print('shown: %d/%d' % (limit, len(active)))
    print('Resources are allocated quantities; task IDs come from JSON without table truncation.')

    section('condor_queue')
    print('query_status:', condor_status)
    worker_jobs = [j for j in jobs or [] if worker_origin(j) is not None]
    current_user = pwd.getpwuid(os.getuid()).pw_name
    if owner == current_user:
        rc, process_out, process_err = run(['pgrep', '-u', str(os.getuid()), '-af', '[w]ork_queue_factory'], timeout)
    else:
        rc, process_out, process_err = None, '', 'OWNER differs from current user'
    factories = [f for f in (factory_from_process(line) for line in process_out.splitlines()) if f] if rc == 0 else []
    factory_scratch = {f['scratch'] for f in factories if f['scratch']}
    if jobs is not None:
        print('owner_jobs=%d identified_wq_jobs=%d other_or_unclassified=%d' % (len(jobs), len(worker_jobs), len(jobs) - len(worker_jobs)))
        counts = collections.Counter(j.get('JobStatus') for j in jobs)
        print(' '.join('%s=%d' % (label, counts[state]) for state, label in
                      [(1,'idle'),(2,'running'),(3,'removing'),(4,'completed'),(5,'held'),(6,'transferring'),(7,'suspended')]))
    section('explicit_cgroup_probes')
    if not cgroup_ids.strip():
        print('DISABLED: CGROUP_JOB_IDS unset')
    elif jobs is None:
        print('UNAVAILABLE: owner Condor snapshot unavailable')
    else:
        probes = cgroup_probes(cgroup_ids, jobs, owner, schedd,
                               max_cgroup_probes, cgroup_probe_timeout)
        if not probes:
            print('<no eligible RUNNING owner-snapshot IDs>')
        for job_id, probe_rc, probe_out, probe_err in probes:
            print('job=%s status=%s' % (job_id, 'OK' if probe_rc == 0 else
                                        'TIMEOUT' if probe_rc == 124 else 'FAILED'))
            print(clean(probe_out).strip() if probe_rc == 0 else probe_err)
    section('condor_worker_profiles')
    print('Scope: identified WQ jobs of this owner across ALL managers; Condor RUNNING is not WQ busy.')
    profiles = collections.defaultdict(collections.Counter)
    origins = collections.Counter()
    for j in worker_jobs:
        profile = resource_profile(j)
        profiles[profile][j.get('JobStatus')] += 1
        origins[worker_origin(j)] += 1
    if jobs is None:
        print('<UNKNOWN>')
    else:
        table(['cores', 'memory_MiB', 'disk_MiB', 'running', 'idle', 'held', 'removing', 'other', 'total'],
              [list(p) + [str(c[s]) for s in (2,1,5,3)] + [str(sum(v for k,v in c.items() if k not in (2,1,5,3))), str(sum(c.values()))]
               for p,c in sorted(profiles.items())])
        for origin, count in sorted(origins.items()):
            print('origin jobs=%d %s' % (count, clean(origin)))

    section('condor_worker_pools')
    print('Scratch equality is provenance correlation, not proof of WQ connection.')
    if jobs is None:
        print('UNAVAILABLE:', condor_status)
    else:
        pools = pool_counts(worker_jobs)
        table(['binding', 'origin', 'selector', 'cores', 'memory_MiB', 'disk_MiB',
               'running', 'idle', 'held', 'removing', 'total'],
              [[('LIVE_FACTORY_BOUND' if key[0] in factory_scratch else 'NO_LIVE_FACTORY_MATCH'),
                key[0], key[1], key[2], key[3], key[4]] +
               [str(counts[state]) for state in (2, 1, 5, 3)] + [str(sum(counts.values()))]
               for key, counts in sorted(pools.items())])

    section('active_task_condor_candidates')
    print('Host-only candidates; no exact worker/job identity or resource-fit claim.')
    if task_status == 'DISABLED':
        print('DISABLED')
    elif tasks is None or jobs is None:
        print('<UNKNOWN>')
    for t in active[:limit]:
        candidates = [j for j in worker_jobs if j.get('JobStatus') == 2 and
                      str(j.get('RemoteHost', '')).partition('@')[2].rstrip('.').lower() == str(t.get('host', '')).rstrip('.').lower()
                      and bool(t.get('host'))]
        print('task=%s candidates=%s' % (t['taskid'], ','.join('%s.%s' % (j.get('ClusterId'),j.get('ProcId')) for j in candidates) or 'unresolved'))

    section('held_jobs')
    held = [j for j in jobs or [] if j.get('JobStatus') == 5]
    if jobs is None:
        print('<UNKNOWN>')
    else:
        table(['job', 'kind', 'origin', 'binding', 'selector', 'memory_MiB', 'usage_MiB', 'last_host', 'reason'],
              [['%s.%s' % (j.get('ClusterId'),j.get('ProcId')), 'WQ' if worker_origin(j) else 'other/unknown',
                *held_job_attribution(j, factory_scratch), fmt(j.get('RequestMemory')),
                fmt(j.get('MemoryUsage')), j.get('LastRemoteHost', '?'), j.get('HoldReason','?')]
               for j in held])

    section('work_queue_factories')
    print('visibility:', 'VISIBLE' if factories else
          'NOT_OBSERVABLE_FOR_SELECTED_OWNER' if owner != current_user else
          'NO_VISIBLE_FACTORY; host PID namespace may be hidden' if rc == 1 else 'UNAVAILABLE')
    table(['pid', 'selector', 'config_path', 'config_status', 'scratch', 'min_workers', 'max_workers',
           'workers_per_cycle', 'worker_timeout', 'factory_timeout', 'cores', 'memory', 'disk', 'active_condor_jobs',
           'running', 'idle', 'held', 'warning'],
          [[f['pid'], f['selector'], f['config_path'] or '?', f['config_status'], f['scratch'] or '?'] +
           [fmt(f['settings'].get(k)) for k in ('min-workers', 'max-workers', 'workers-per-cycle',
                                             'worker_timeout', 'factory_timeout', 'cores', 'memory', 'disk')] +
           ([str(sum(counts.values())), str(counts[2]), str(counts[1]), str(counts[5]),
             warning or '-'] if jobs is not None else ['?', '?', '?', '?', 'CONDOR_UNAVAILABLE'])
           for f in factories for counts, warning in [factory_warnings(f, worker_jobs)]])

    section('recent_worker_exits')
    print('query_status:', history_status)
    print('lookback_seconds=%d result_cap=%d scan_cap=%d; ExitCode is evidence, not a diagnosis.' %
          (lookback, history_max, history_max * 10))
    if history is None:
        print('UNAVAILABLE')
    else:
        exits = recent_exits(history)
        table(['origin', 'recent_exits', 'exit0', 'exit_nonzero', 'signaled',
               'median_runtime_s', 'top_exit_codes'],
              [[origin, str(row[0]), str(row[1]), str(row[2]), str(row[3]),
                fmt(row[4]), ','.join('%s:%d' % (code, count) for code, count in row[5].most_common(3)) or '?']
               for origin, row in sorted(exits.items())])
        if history_raw_count >= history_max:
            print('RESULT_CAP_REACHED: counts may be incomplete within the lookback.')
        print('Scan cap may also truncate older records within the lookback.')

    section('log_progress')
    progressing = False
    if not logdir:
        print('LOGDIR unset; no trend inferred from a single snapshot.')
    else:
        print('explicit log source:', clean(logdir))
        if log_sample:
            last, first, stamp = log_sample
            age = time.time() - stamp
            fresh = log_fresh and -5 <= age <= age_limit
            print('sample=%s age=%.1fs %s' % (datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).isoformat(), age, 'FRESH' if fresh else 'STALE/clock mismatch'))
            if log_error:
                print(log_error + '; not used for health')
            fields = ('tasks_submitted', 'tasks_done', 'tasks_waiting', 'tasks_running', 'tasks_failed', 'tasks_exhausted_attempts', 'bytes_sent', 'bytes_received')
            deltas = {}
            if first:
                dt = (float(last['timestamp']) - float(first['timestamp'])) / 1e6
                print('window=%.1fs' % dt)
                for k in fields:
                    a,b = number(first.get(k)),number(last.get(k))
                    if a is not None and b is not None:
                        deltas[k] = b-a
            table(['counter', 'latest', 'delta'], [[k,fmt(last.get(k)), '%+g' % deltas[k] if k in deltas else '?'] for k in fields])
            monotonic_keys = ('tasks_submitted','tasks_done','bytes_sent','bytes_received')
            reset = any(deltas.get(k,0) < 0 for k in monotonic_keys)
            progressing = fresh and not reset and any(deltas.get(k,0) > 0 for k in ('tasks_done','bytes_sent','bytes_received'))
            if reset:
                print('counter reset: trend not used for health')
            print('Task deltas include all categories; accumulation estimates and event totals are distinct.')
        else:
            print('unavailable:', log_error)

    section('selected_manager_health')
    if queue is None:
        if progressing:
            print('status: PROGRESS_IN_LOGS / DIRECT_UNKNOWN')
        elif tasks is not None and active:
            print('status: ACTIVE_TASKS_SNAPSHOT / SUMMARY_UNKNOWN')
        elif log_sample and log_fresh:
            print('status: LOG_SNAPSHOT / DIRECT_UNKNOWN')
        elif catalog_fresh:
            print('status: CATALOG_SNAPSHOT / DIRECT_UNKNOWN')
        else:
            print('status: UNKNOWN')
        print('reason: direct summary unavailable; fallback source and age are shown above')
        print('No zero-task, deadlock or completion conclusion from a timeout.')
    elif progressing:
        print('status: PROGRESSING (recent completions and/or data transfers)')
    elif number(queue.get('tasks_running')) > 0 or (number(queue.get('tasks_on_workers')) or 0) > 0:
        print('status: ACTIVE_SNAPSHOT (progress over time not established)')
    elif number(queue.get('tasks_waiting')) > 0:
        print('status: WAITING (queued work, no activity reported in summary; cause unproven)')
    elif number(queue.get('tasks_with_results')) == 0:
        print('status: QUIET_SNAPSHOT (not proof of production completion)')
    else:
        print('status: RESULTS_PENDING_OR_UNKNOWN')
    if tasks is None:
        print('task_details:', task_status)
    if jobs is None:
        print('condor_details:', condor_status)
    if project is not None and selection_status != 'OK':
        print('selection_alert:', selection_status)

    section('owner_wide_manager_alerts')
    if catalog is None:
        print('UNAVAILABLE:', catalog_status)
    else:
        alerts = manager_alerts(owner_managers)
        table(['project', 'alert'], [[r.get('project', '?'), 'WAITING_WITH_ZERO_CONNECTED_WORKERS'] for r in alerts])
        if project is not None and selection_status != 'OK':
            print('selected_project:', selection_status)

    section('worker_supply_alerts')
    if jobs is None:
        print('condor: UNAVAILABLE')
    else:
        if held:
            print('held_owner_jobs=%d; see held_jobs for worker/nonworker scope' % len(held))
    if history is None:
        print('recent_history: UNAVAILABLE')
    if jobs is not None:
        for kind, origin, selector, count in supply_alerts(worker_jobs, history, factories):
            print('%s count_or_pid=%s origin=%s selector=%s' %
                  (kind, count, clean(origin), clean(selector)))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OverflowError) as exc:
        print('dashboard configuration/data error:', clean(exc), file=sys.stderr)
        sys.exit(2)
