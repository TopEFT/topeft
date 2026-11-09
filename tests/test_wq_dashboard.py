"""Focused, scheduler-free coverage for the read-only operator dashboard."""
import contextlib
import getpass
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / 'analysis' / 'topeft_run2' / 'wq_dashboard.py'
spec = importlib.util.spec_from_file_location('wq_dashboard', SOURCE)
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def worker(origin='/scratch/a', selector='^project-.*', status=2, cores=2):
    return {'Owner': 'operator', 'Cmd': origin + '/condor.sh',
            'Args': './with_oasis_certs ./work_queue_worker -M ' + selector, 'JobStatus': status,
            'RequestCpus': cores, 'RequestMemory': 8000, 'RequestDisk': 8192000}


class DashboardTests(unittest.TestCase):
    def test_catalog_owner_normalization_and_freshness(self):
        now = time.time()
        records = [{'owner': 'operator', 'project': 'a', 'name': 'h', 'port': 1,
                    'lastheardfrom': now - 2, 'tasks_complete': 3, 'workers': 4},
                   {'owner': 'operator', 'project': 'b', 'name': 'h', 'port': 2,
                    'lastheardfrom': now - 300},
                   {'project': 'unknown', 'name': 'h', 'port': 3},
                   {'owner': 'someone_else', 'project': 'foreign', 'name': 'h', 'port': 4}]
        rows = dashboard.owner_catalog(records, 'operator', now, 120)
        self.assertEqual([r['project'] for r in rows], ['a', 'b'])
        self.assertEqual(rows[0]['tasks_done'], 3)
        self.assertEqual(rows[0]['workers_connected'], 4)
        self.assertEqual([r['_fresh'] for r in rows], [True, False])
        self.assertEqual(dashboard.matching_catalog(records, 'h', 3, 'operator'), [])
        near_expiry = [dict(records[0], lastheardfrom=now - 119)]
        self.assertTrue(dashboard.owner_catalog(near_expiry, 'operator', now, 120)[0]['_fresh'])
        self.assertFalse(dashboard.owner_catalog(near_expiry, 'operator', now + 10, 120)[0]['_fresh'])

    def test_project_selection_exact_unique_zero_ambiguous(self):
        now = time.time()
        records = [{'owner': 'operator', 'project': 'exact', 'name': 'h1', 'port': 9,
                    'lastheardfrom': now},
                   {'owner': 'operator', 'project': 'exact-ish', 'name': 'h2', 'port': 10,
                    'lastheardfrom': now}]
        rows = dashboard.owner_catalog(records, 'operator', now, 120)
        self.assertEqual(dashboard.select_project(rows, 'exact'), (('h1', '9'), 'OK'))
        self.assertEqual(dashboard.select_project(rows, 'ex'), (None, 'NO_FRESH_MATCH'))
        rows.extend(dashboard.owner_catalog([dict(records[0], name='h3', port=11)], 'operator', now, 120))
        self.assertEqual(dashboard.select_project(rows, 'exact')[1], 'AMBIGUOUS_FRESH_MATCH')
        rows[0]['_fresh'] = False
        self.assertEqual(dashboard.select_project(rows, 'exact'), (('h3', '11'), 'OK'))

    def test_task_enumeration_disabled_makes_no_task_call(self):
        calls = []
        def query(args, timeout):
            calls.append(args)
            return [{'tasks_waiting': 2, 'tasks_running': 0}], 'OK'
        summary, status, tasks, task_status = dashboard.manager_queries('host', '9164', 5, False, 5, query)
        self.assertEqual((status, task_status, tasks), ('OK', 'DISABLED', None))
        self.assertEqual(len(calls), 1)
        self.assertNotIn('-T', calls[0])
        dashboard.manager_queries('host', '9164', 5, True, 7, query)
        self.assertEqual(len(calls), 3)
        self.assertIn('-T', calls[2])
        self.assertIn('7', calls[2])

    def test_project_identity_is_rechecked_before_detail_or_tasks(self):
        calls = []
        def wrong_project(args, timeout):
            calls.append(args)
            return [{'project': 'other', 'tasks_waiting': 1, 'tasks_running': 0}], 'OK'
        summary, status, tasks, task_status = dashboard.manager_queries(
            'host', '9164', 5, True, 7, wrong_project, expected_project='wanted')
        self.assertEqual((summary, status, tasks, task_status),
                         (None, 'DIRECT_PROJECT_MISMATCH', None, 'SKIPPED_DIRECT_PROJECT_MISMATCH'))
        self.assertEqual(len(calls), 1)
        self.assertNotIn('-T', calls[0])
        def timed_out(args, timeout):
            return None, 'TIMEOUT'
        self.assertEqual(dashboard.manager_queries('host', '9164', 5, True, 7, timed_out,
                                                   expected_project='wanted')[3],
                         'SKIPPED_UNVERIFIED_DIRECT_PROJECT')
        def correct_project(args, timeout):
            calls.append(args)
            return ([{'project': 'wanted', 'tasks_waiting': 1, 'tasks_running': 0}], 'OK')
        dashboard.manager_queries('host', '9164', 5, True, 7, correct_project,
                                  expected_project='wanted')
        self.assertIn('-T', calls[-1])

    def test_worker_origin_selector_and_pool_identity(self):
        a = worker()
        b = worker(selector='other')
        c = worker(cores=4)
        d = worker(origin='/scratch/b', status=5)
        self.assertEqual(dashboard.worker_origin(a), '/scratch/a')
        self.assertEqual(dashboard.worker_selector(a), '^project-.*')
        self.assertEqual(dashboard.worker_selector({'Args': './work_queue_worker'}), '?')
        self.assertEqual(len(dashboard.pool_counts([a, b, c, d])), 4)
        self.assertEqual(dashboard.pool_counts([d])[dashboard.pool_key(d)][5], 1)
        self.assertIsNone(dashboard.worker_origin({'Cmd': '/usr/bin/python', 'Args': 'analysis.py'}))
        self.assertIsNone(dashboard.worker_origin({'Cmd': '/bin/bash',
                                                   'Args': "-c 'echo work_queue_worker'"}))

    def test_worker_invocation_is_independent_of_scratch_name(self):
        generic = worker(origin='/tmp/wq-apiccine-tau')
        self.assertEqual(dashboard.worker_origin(generic), '/tmp/wq-apiccine-tau')
        misleading = {'Cmd': '/tmp/wq-apiccine-factory-misc/condor.sh',
                      'Args': '/bin/echo hello'}
        self.assertIsNone(dashboard.worker_origin(misleading))
        self.assertEqual(dashboard.worker_origin({'Cmd': '/some/path/work_queue_worker'}),
                         '/some/path')
        self.assertEqual(dashboard.worker_origin({'Cmd': '/some/path/work_queue_worker',
                                                  'Args': "'malformed"}), '/some/path')
        self.assertEqual(dashboard.worker_origin({'Cmd': generic['Cmd'],
                                                  'Arguments': './work_queue_worker -M manager'}),
                         '/tmp/wq-apiccine-tau')
        accepted_wrapper = {'Cmd': '/tmp/wq-apiccine-manual/with_oasis_certs',
                            'Args': './work_queue_worker -M manager'}
        self.assertEqual(dashboard.worker_origin(accepted_wrapper), '/tmp/wq-apiccine-manual')
        self.assertEqual(dashboard.worker_selector(accepted_wrapper), 'manager')
        self.assertIsNone(dashboard.worker_origin({'Cmd': accepted_wrapper['Cmd'],
                                                   'Args': '/bin/echo ./work_queue_worker'}))
        self.assertIsNone(dashboard.worker_origin({'Cmd': accepted_wrapper['Cmd'],
                                                   'Args': "'./work_queue_worker"}))
        fallback = {'Cmd': generic['Cmd'], 'Args': '', 'Arguments': './work_queue_worker -M fallback'}
        self.assertEqual(dashboard.worker_origin(fallback), '/tmp/wq-apiccine-tau')
        self.assertEqual(dashboard.worker_selector(fallback), 'fallback')
        self.assertIsNone(dashboard.worker_origin({'Cmd': generic['Cmd'],
                                                   'Args': "'./with_oasis_certs ./work_queue_worker"}))
        self.assertIsNone(dashboard.worker_origin({'Cmd': generic['Cmd'],
                                                   'Args': '/bin/echo ./work_queue_worker'}))

    def test_factory_argv_config_and_correlation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'factory.json'
            path.write_text(json.dumps({'manager-name': 'from-json', 'max-workers': 1,
                                        'min-workers': 0, 'scratch-dir': '/scratch/a',
                                        'timeout': 900}))
            parsed = dashboard.factory_from_process('123 /bin/work_queue_factory -C %s -S /scratch/a' % path)
            self.assertEqual(parsed['selector'], 'from-json')
            self.assertEqual(parsed['config_status'], 'OK')
            self.assertEqual(parsed['settings']['worker_timeout'], 900)
            self.assertIsNone(parsed['settings']['factory_timeout'])
            distinct = dashboard.factory_from_process(
                '124 /bin/work_queue_factory -C %s --factory-timeout 0' % path)
            self.assertEqual(distinct['settings']['worker_timeout'], 900)
            self.assertEqual(distinct['settings']['factory_timeout'], '0')
            cli_only = dashboard.factory_from_process('125 /bin/work_queue_factory --factory-timeout 0')
            self.assertIsNone(cli_only['settings']['worker_timeout'])
            self.assertEqual(cli_only['settings']['factory_timeout'], '0')
            self.assertEqual(dashboard.factory_warnings(parsed, [worker(), worker(status=1)])[1],
                             'OBSERVED_ACTIVE_JOBS_ABOVE_DECLARED_MAX')
            self.assertIsNone(dashboard.factory_warnings(parsed, [worker(origin='/scratch/b')])[1])
            cli = dashboard.factory_from_process('124 /bin/work_queue_factory -M cli -C %s' % path)
            self.assertEqual(cli['selector'], 'cli')
            bad = Path(directory) / 'bad.json'
            bad.write_text('{bad')
            self.assertTrue(dashboard.factory_from_process('125 /bin/work_queue_factory -C %s' % bad)
                            ['config_status'].startswith('UNAVAILABLE'))
            self.assertTrue(dashboard.factory_from_process('126 /bin/work_queue_factory -C %s' %
                             (Path(directory) / 'missing'))['config_status'].startswith('UNAVAILABLE'))

    def test_cgroup_probe_requires_explicit_running_owner_snapshot_id(self):
        jobs = [dict(worker(), ClusterId=12, ProcId=0),
                dict(worker(status=1), ClusterId=13, ProcId=0),
                dict(worker(), Owner='other', ClusterId=14, ProcId=0),
                dict(worker(), ClusterId=15, ProcId=0)]
        calls = []
        def fake_run(args, timeout, input_text=None):
            calls.append((args, timeout, input_text))
            return 0, 'memory.current: 123\n', ''
        self.assertEqual(dashboard.cgroup_probes('', jobs, 'operator', 'schedd', 4, 8, fake_run), [])
        self.assertEqual(calls, [])
        rows = dashboard.cgroup_probes('bad 99.0,13.0 14.0 12.0 12.0', jobs,
                                      'operator', 'schedd', 4, 8, fake_run)
        self.assertEqual([row[0] for row in rows], ['12.0'])
        self.assertEqual(calls[0][:2],
                         (['condor_ssh_to_job', '-name', 'schedd', '12.0', 'bash', '-s'], 8))
        script = calls[0][2]
        for name in ('_CONDOR_JOB_PIDS', 'memory.current', 'memory.peak',
                     'memory.max', 'memory.events', '/proc/$pid/cgroup'):
            self.assertIn(name, script)
        self.assertEqual(dashboard.cgroup_probes('12.0', jobs, 'operator', 'schedd', 0, 8, fake_run), [])
        self.assertEqual(len(calls), 1)
        capped = dashboard.cgroup_probes('12.0 15.0', jobs, 'operator', 'schedd', 1, 8, fake_run)
        self.assertEqual([row[0] for row in capped], ['12.0'])
        self.assertEqual(len(calls), 2)

    def test_cgroup_script_is_piped_through_bounded_subprocess(self):
        process = mock.Mock(returncode=0)
        process.communicate.return_value = ('memory.current: 123\n', '')
        with mock.patch.object(dashboard.subprocess, 'Popen', return_value=process) as popen:
            self.assertEqual(dashboard.run(['condor_ssh_to_job', 'bash', '-s'], 8,
                                           input_text=dashboard.CGROUP_READ_SCRIPT),
                             (0, 'memory.current: 123\n', ''))
        self.assertEqual(popen.call_args.kwargs['stdin'], dashboard.subprocess.PIPE)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        process.communicate.assert_called_once_with(input=dashboard.CGROUP_READ_SCRIPT, timeout=8)

    def test_cgroup_probe_opt_out_and_rendered_timeout_labels(self):
        def query(args, timeout):
            if args[0] == 'condor_q':
                return [dict(worker(), ClusterId=12, ProcId=0)], 'OK'
            if args[0] == 'condor_history':
                return [], 'OK'
            if args[-2:] == ['host', '9164']:
                return [{'tasks_waiting': 0, 'tasks_running': 0}], 'OK'
            return [], 'OK'
        factory = {'pid': '1', 'selector': '?', 'config_path': None, 'config_status': 'OK',
                   'scratch': '/scratch/a', 'settings': {'worker_timeout': 900,
                                                       'factory_timeout': '0'}}
        output = io.StringIO()
        with mock.patch.object(dashboard, 'query', side_effect=query), \
             mock.patch.object(dashboard, 'run', return_value=(0, '1 work_queue_factory\n', '')) as run_mock, \
             mock.patch.object(dashboard.pwd, 'getpwuid', return_value=mock.Mock(pw_name='operator')), \
             mock.patch.object(dashboard, 'factory_from_process', return_value=factory), \
             mock.patch.object(sys, 'argv', ['wq_dashboard.py']), \
             mock.patch.dict(os.environ, {'OWNER': 'operator', 'MANAGER_HOST': 'host',
                                          'MANAGER_PORT': '9164'}, clear=True), \
             contextlib.redirect_stdout(output):
            dashboard.main()
        self.assertIn('DISABLED: CGROUP_JOB_IDS unset', output.getvalue())
        self.assertIn('worker_timeout', output.getvalue())
        self.assertIn('factory_timeout', output.getvalue())
        self.assertRegex(output.getvalue(), r'900\s+0')
        self.assertFalse(any(call.args[0][0] == 'condor_ssh_to_job' for call in run_mock.call_args_list))

    def test_help_exits_before_any_queries_or_probes(self):
        output = io.StringIO()
        with mock.patch.object(dashboard, 'query') as query_mock, \
             mock.patch.object(dashboard, 'run') as run_mock, \
             mock.patch.object(sys, 'argv', ['wq_dashboard.py', '--help']), \
             mock.patch.dict(os.environ, {'CGROUP_JOB_IDS': '12.0'}, clear=True), \
             contextlib.redirect_stdout(output):
            dashboard.main()
        query_mock.assert_not_called()
        run_mock.assert_not_called()
        self.assertIn('CGROUP_JOB_IDS', output.getvalue())
        self.assertIn('Host/task matching alone does not authorize', output.getvalue())

    def test_recent_exits_and_missing_fields(self):
        rows = [dict(worker(), ExitCode=0, RemoteWallClockTime=10),
                dict(worker(), ExitCode=2, RemoteWallClockTime=20),
                dict(worker(), ExitBySignal=True, ExitSignal=9, RemoteWallClockTime=30),
                worker(), {'Cmd': '/bin/other', 'ExitCode': 1},
                {'Cmd': '/bin/bash', 'Args': "-c 'echo work_queue_worker'", 'ExitCode': 2}]
        exits = dashboard.recent_exits(rows)['/scratch/a']
        self.assertEqual(exits[:4], (4, 1, 1, 1))
        self.assertEqual(exits[4], 20)
        self.assertEqual(exits[5][2], 1)
        self.assertEqual(dashboard.recent_exits(None), {})

    def test_history_failure_is_not_zero(self):
        with mock.patch.object(dashboard, 'run', return_value=(124, '', 'query deadline exceeded')):
            records, status = dashboard.query(['condor_history'], 3)
        self.assertIsNone(records)
        self.assertEqual(status, 'TIMEOUT')

    def test_rendered_history_unavailable_and_default_task_opt_out(self):
        calls = []
        history_timeouts = []
        def query(args, timeout):
            calls.append(args)
            if args[0] == 'condor_history':
                history_timeouts.append(timeout)
                return None, 'TIMEOUT'
            if args[0] == 'condor_q':
                return [], 'OK'
            if args[-2:] == ['host', '9164']:
                return [{'tasks_waiting': 0, 'tasks_running': 0,
                         'tasks_with_results': 0}], 'OK'
            return [], 'OK'
        output = io.StringIO()
        with mock.patch.object(dashboard, 'query', side_effect=query), \
             mock.patch.object(dashboard, 'run', return_value=(1, '', '')) as run_mock, \
             mock.patch.object(sys, 'argv', ['wq_dashboard.py']), \
             mock.patch.dict(os.environ, {'OWNER': 'operator', 'MANAGER_HOST': 'host',
                                          'MANAGER_PORT': '9164'}, clear=True), \
             contextlib.redirect_stdout(output):
            dashboard.main()
        self.assertFalse(any('-T' in args for args in calls))
        history_args = [args for args in calls if args[0] == 'condor_history']
        self.assertEqual(len(history_args), 1)
        self.assertIn('-scanlimit', history_args[0])
        self.assertEqual(history_timeouts, [20.0])
        if getpass.getuser() != 'operator':
            run_mock.assert_not_called()
            self.assertIn('NOT_OBSERVABLE_FOR_SELECTED_OWNER', output.getvalue())
        self.assertIn('=== recent_worker_exits ===\nquery_status: TIMEOUT', output.getvalue())
        self.assertIn('recent_history: UNAVAILABLE', output.getvalue())
        self.assertIn('=== active_work_queue_tasks ===\nquery_status: DISABLED', output.getvalue())

    def test_project_mismatch_keeps_owner_sections_and_skips_tasks(self):
        calls = []
        def query(args, timeout):
            calls.append(args)
            if args[0] == 'condor_history':
                return [], 'OK'
            if args[0] == 'condor_q':
                return [], 'OK'
            if args[-2:] == ['host', '9164']:
                return [{'project': 'other', 'tasks_waiting': 1, 'tasks_running': 0}], 'OK'
            return [{'owner': 'operator', 'project': 'wanted', 'name': 'host',
                     'port': 9164, 'lastheardfrom': time.time()}], 'OK'
        output = io.StringIO()
        with mock.patch.object(dashboard, 'query', side_effect=query), \
             mock.patch.object(dashboard, 'run', return_value=(1, '', '')), \
             mock.patch.object(sys, 'argv', ['wq_dashboard.py']), \
             mock.patch.dict(os.environ, {'OWNER': 'operator', 'MANAGER_PROJECT': 'wanted',
                                          'TASK_DETAILS': '1'}, clear=True), \
             contextlib.redirect_stdout(output):
            dashboard.main()
        self.assertFalse(any('-T' in args for args in calls))
        self.assertIn('owner_records=1', output.getvalue())
        self.assertIn('selection: DIRECT_PROJECT_MISMATCH', output.getvalue())
        self.assertIn('query_status: SKIPPED_DIRECT_PROJECT_MISMATCH', output.getvalue())

    def test_held_and_health_alerts(self):
        held = worker(status=5)
        factory = {'scratch': '/scratch/a', 'selector': 'selector', 'pid': '2',
                   'settings': {'max-workers': 0}}
        alerts = dashboard.supply_alerts([held], [dict(worker(), ExitCode=2)], [factory])
        self.assertEqual({a[0] for a in alerts},
                         {'HELD_WORKER_JOBS', 'RECENT_NONZERO_OR_SIGNALED',
                          'OBSERVED_ACTIVE_JOBS_ABOVE_DECLARED_MAX'})
        self.assertEqual(dashboard.held_job_attribution(held, {'/scratch/a'}),
                         ('/scratch/a', 'LIVE_FACTORY_BOUND', '^project-.*'))
        self.assertEqual(dashboard.held_job_attribution(held, set())[1], 'NO_LIVE_FACTORY_MATCH')
        rows = dashboard.owner_catalog([{'owner': 'operator', 'project': 'p',
                                         'tasks_waiting': 5, 'workers': 0,
                                         'lastheardfrom': time.time()}],
                                       'operator', time.time(), 120)
        self.assertEqual(len(dashboard.manager_alerts(rows)), 1)
        rows[0]['_fresh'] = False
        self.assertEqual(dashboard.manager_alerts(rows), [])

    def test_sanitization_and_summary_semantics(self):
        self.assertEqual(dashboard.clean('a\x1bb\n'), 'a b ')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            dashboard.table(['value'], [['bad\nrow']])
        self.assertNotIn('bad\nrow', output.getvalue())
        self.assertEqual(dashboard.normalized_summary({'tasks_complete': 7})['tasks_done'], 7)


class AutodiscoveryTests(unittest.TestCase):
    now = 1800000000

    def record(self, project='project-a', port=1026, **values):
        return dict({'owner': 'operator', 'project': project, 'name': 'fixture-host',
                     'port': port, 'lastheardfrom': self.now, 'tasks_waiting': 3,
                     'tasks_running': 0, 'workers': 0}, **values)

    def summary(self, project, **values):
        return dict({'project': project, 'tasks_waiting': 3, 'tasks_running': 1,
                     'categories': [{'category': 'category-' + project,
                                     'tasks_waiting': 3, 'tasks_running': 1}]}, **values)

    def invoke(self, records, summaries=None, env=None, jobs=None, catalog_status='OK',
               direct_fn=None):
        calls = []
        summaries = summaries or {}
        def fake_query(args, timeout):
            calls.append((args, timeout))
            if args[0] == 'condor_q':
                return jobs or [], 'OK'
            if args[0] == 'condor_history':
                return [], 'OK'
            if args[0] == 'work_queue_status' and len(args) == 4:
                return records, catalog_status
            if args[0] == 'work_queue_status' and '-T' not in args:
                if direct_fn:
                    return direct_fn(args, timeout)
                return summaries[tuple(args[-2:])]
            raise AssertionError('unexpected service call: %r' % args)
        output = io.StringIO()
        with mock.patch.object(dashboard, 'query', side_effect=fake_query), \
             mock.patch.object(dashboard, 'run', return_value=(1, '', '')) as run_mock, \
             mock.patch.object(dashboard.subprocess, 'Popen', side_effect=AssertionError('live process')), \
             mock.patch.object(dashboard, 'stats_tail', side_effect=AssertionError('unbound logs')) as logs, \
             mock.patch.object(dashboard.getpass, 'getuser', return_value='operator'), \
             mock.patch.object(dashboard.pwd, 'getpwuid', return_value=mock.Mock(pw_name='operator')), \
             mock.patch.object(dashboard.time, 'time', return_value=self.now), \
             mock.patch.object(sys, 'argv', ['wq_dashboard.py']), \
             mock.patch.dict(os.environ, env or {}, clear=True), \
             contextlib.redirect_stdout(output):
            dashboard.main()
        return output.getvalue(), calls, run_mock.call_args_list, logs.call_count

    def endpoints(self, calls):
        return [tuple(args[-2:]) for args, _ in calls
                if args[0] == 'work_queue_status' and len(args) > 4]

    def test_unconfigured_two_managers_and_single_owner_population(self):
        records = [self.record('run2-attempt1'), self.record('run3-attempt2', 1027)]
        summaries = {('fixture-host', str(r['port'])): ([self.summary(r['project'])], 'OK')
                     for r in records}
        jobs = [worker(), worker(origin='/scratch/topup', cores=4),
                worker(status=5), {'Owner': 'operator', 'Cmd': '/bin/other', 'JobStatus': 2}]
        text, calls, processes, _ = self.invoke(records, summaries, jobs=jobs)
        self.assertCountEqual(self.endpoints(calls), list(summaries))
        self.assertNotIn('9164', text)
        self.assertIn('AUTO_OWNER_DISCOVERY', text)
        for r in records:
            self.assertIn('category-' + r['project'], text)
            self.assertIn('project=' + r['project'] + ' endpoint=fixture-host:' + str(r['port']), text)
        self.assertEqual(text.count('summary_source: DIRECT'), 2)
        self.assertEqual(text.count('owner_jobs=4 identified_wq_jobs=3 other_or_unclassified=1'), 1)
        for section in ('condor_queue', 'condor_worker_profiles', 'condor_worker_pools',
                        'held_jobs', 'work_queue_factories', 'recent_worker_exits', 'worker_supply_alerts'):
            self.assertEqual(text.count('=== ' + section + ' ==='), 1)
        self.assertIn('/scratch/topup', text)
        for command in ('condor_q', 'condor_history'):
            self.assertEqual(sum(args[0] == command for args, _ in calls), 1)
        self.assertEqual(len(processes), 1)
        self.assertEqual(processes[0].args[0][0], 'pgrep')
        self.assertFalse(any('-T' in args for args, _ in calls))

    def test_next_invocation_rediscovers_port_and_attempt(self):
        for project, port in [('run2-attempt1', 1026), ('run2-attempt3', 1042)]:
            with self.subTest(project=project):
                text, calls, _, _ = self.invoke([self.record(project, port)],
                    {('fixture-host', str(port)): ([self.summary(project)], 'OK')})
                self.assertEqual(self.endpoints(calls), [('fixture-host', str(port))])
                self.assertIn('category-' + project, text)

    def test_single_zero_worker_and_empty_catalog(self):
        text, calls, _, _ = self.invoke([self.record()],
            {('fixture-host', '1026'): ([self.summary('project-a', workers=0)], 'OK')})
        self.assertEqual(self.endpoints(calls), [('fixture-host', '1026')])
        self.assertIn('WAITING_WITH_ZERO_CONNECTED_WORKERS', text)
        empty, calls, _, _ = self.invoke([])
        self.assertEqual(self.endpoints(calls), [])
        self.assertIn('EMPTY_OWNER_CATALOG', empty)

    def test_owner_and_incomplete_project_records(self):
        records = [self.record(owner='foreign'), self.record(owner=None),
                   self.record(project=None), self.record(project='  '), self.record(project=['bad'])]
        text, calls, _, _ = self.invoke(records)
        self.assertEqual(self.endpoints(calls), [])
        self.assertIn('owner_records=3 omitted_missing_or_other_owner=2', text)
        self.assertIn('INVALID_PROJECT', text)
        self.assertIn("['bad']", text)

    def test_unsafe_or_missing_endpoints_never_queried(self):
        invalid = [{'port': value} for value in (True, False, 0, -1, 65536, 1026.5, '2.5', None)]
        invalid += [{'name': host} for host in ('', '-option', 'white space', 'tab\thost', 'bad\x00host')]
        invalid += [{'name': None}, {'port': 'not-a-port'}]
        for values in invalid:
            with self.subTest(values=values):
                text, calls, _, _ = self.invoke([self.record(**values)])
                self.assertEqual(self.endpoints(calls), [])
                self.assertIn('FRESH_RECORD_WITHOUT_ENDPOINT', text)
                self.assertIn('summary_source: CATALOG', text)
        for port in (1, 65535, '1026'):
            with self.subTest(valid_port=port):
                _, calls, _, _ = self.invoke([self.record(port=port)],
                    {('fixture-host', str(port)): ([self.summary('project-a')], 'OK')})
                self.assertEqual(self.endpoints(calls), [('fixture-host', str(port))])

    def test_stale_unknown_and_fresh_have_separate_sources(self):
        records = [self.record('stale', 1028, lastheardfrom=self.now - 300),
                   self.record('unknown', 1029, lastheardfrom=None), self.record()]
        text, calls, _, _ = self.invoke(records,
            {('fixture-host', '1026'): ([self.summary('project-a')], 'OK')})
        self.assertEqual(self.endpoints(calls), [('fixture-host', '1026')])
        self.assertEqual(text.count('summary_source: DIRECT'), 1)
        self.assertIn('CATALOG age=300s STALE/UNKNOWN_AGE', text)
        self.assertIn('CATALOG age=unknown STALE/UNKNOWN_AGE', text)

    def test_distinct_attempts_both_queried(self):
        records = [self.record('same-unit-attempt9', 1027), self.record('same-unit-attempt1')]
        text, calls, _, _ = self.invoke(records,
            {('fixture-host', str(r['port'])): ([self.summary(r['project'])], 'OK') for r in records})
        self.assertEqual(len(self.endpoints(calls)), 2)
        for r in records:
            self.assertIn('category-' + r['project'], text)

    def test_duplicate_fresh_project_is_ambiguous_even_with_invalid_endpoint(self):
        for second_port in (1027, None, 1026):
            with self.subTest(second_port=second_port):
                text, calls, _, _ = self.invoke([self.record(), self.record(port=second_port)])
                self.assertEqual(self.endpoints(calls), [])
                self.assertIn('AMBIGUOUS_FRESH_MATCH', text)
                self.assertEqual(text.count('summary_source: CATALOG'), 2)

    def test_direct_failures_are_isolated_and_identity_checked(self):
        failures = [(None, 'TIMEOUT'), ([], 'OK'),
                    ([self.summary('wrong-project')], 'OK'),
                    ([self.summary('bad', owner='someone-else')], 'OK'),
                    ([self.summary('bad', tasks_waiting='malformed')], 'OK')]
        statuses = ['TIMEOUT', 'INVALID_RESPONSE', 'DIRECT_PROJECT_MISMATCH',
                    'DIRECT_OWNER_MISMATCH', 'INVALID_RESPONSE']
        for response, status in zip(failures, statuses):
            with self.subTest(status=status, response=response):
                text, calls, _, _ = self.invoke([self.record('bad'), self.record('good', 1027)],
                    {('fixture-host', '1026'): response,
                     ('fixture-host', '1027'): ([self.summary('good')], 'OK')})
                self.assertEqual(len(self.endpoints(calls)), 2)
                self.assertEqual(text.count('summary_source: DIRECT'), 1)
                self.assertIn('category-good', text)
                self.assertNotIn('category-bad', text)
                self.assertNotIn('category-wrong-project', text)
                self.assertIn('direct_query: ' + status, text)
                self.assertIn('summary_source: CATALOG', text)
                self.assertIn('No zero-task, deadlock or completion conclusion from a timeout.', text)
                self.assertEqual(text.count('=== condor_queue ==='), 1)

    def test_empty_and_failed_catalog_are_distinct_without_legacy_fallback(self):
        empty, empty_calls, _, _ = self.invoke([])
        failed, failed_calls, _, _ = self.invoke(None, catalog_status='TIMEOUT')
        self.assertIn('EMPTY_OWNER_CATALOG', empty)
        self.assertIn('CATALOG_UNAVAILABLE', failed)
        self.assertNotIn('EMPTY_OWNER_CATALOG', failed)
        self.assertEqual(self.endpoints(empty_calls) + self.endpoints(failed_calls), [])

    def test_summary_budget_and_concurrency_are_bounded_and_deterministic(self):
        records = [self.record('project-%d' % i, 1100 + i) for i in reversed(range(7))]
        lock = threading.Lock()
        barrier = threading.Barrier(2)
        in_flight, peaks = [0], []
        def direct(args, timeout):
            with lock:
                in_flight[0] += 1
                peaks.append(in_flight[0])
            try:
                barrier.wait(timeout=5)
                self.assertEqual(timeout, 10)
                project = 'project-%d' % (int(args[-1]) - 1100)
                return [self.summary(project)], 'OK'
            finally:
                with lock:
                    in_flight[0] -= 1
        text, calls, _, _ = self.invoke(records, direct_fn=direct)
        self.assertCountEqual(self.endpoints(calls), [('fixture-host', str(1100 + i)) for i in range(4)])
        self.assertEqual(max(peaks), 2)
        self.assertEqual(text.count('summary_source: DIRECT'), 4)
        self.assertIn('SUMMARY_BUDGET_EXCEEDED', text)
        for i in range(7):
            self.assertIn('project=project-%d endpoint=fixture-host:%d' % (i, 1100 + i), text)
        positions = [text.index('project=project-%d endpoint=' % i) for i in range(7)]
        self.assertEqual(positions, sorted(positions))

    def test_task_and_log_inputs_unbound_without_explicit_manager(self):
        for records in ([], [self.record()]):
            with self.subTest(record_count=len(records)):
                text, calls, processes, log_calls = self.invoke(records,
                    {('fixture-host', '1026'): ([self.summary('project-a')], 'OK')},
                    env={'TASK_DETAILS': '1', 'LOGDIR': '/unbound/logs'})
                self.assertIn('NO_EXPLICIT_DETAIL_TARGET', text)
                self.assertIn('LOGDIR UNBOUND', text)
                self.assertEqual(log_calls, 0)
                self.assertFalse(any('-T' in args for args, _ in calls))
                self.assertFalse(any(c.args[0][0] == 'condor_ssh_to_job' for c in processes))

    def test_explicit_project_precedes_endpoint_and_failure_never_switches(self):
        records = [self.record('wanted'), self.record('other', 1027)]
        env = {'MANAGER_PROJECT': 'wanted', 'MANAGER_HOST': 'ignored', 'MANAGER_PORT': 'invalid'}
        text, calls, _, _ = self.invoke(records,
            {('fixture-host', '1026'): ([self.summary('wanted')], 'OK')}, env=env)
        self.assertEqual(self.endpoints(calls), [('fixture-host', '1026')])
        self.assertIn('EXPLICIT_PROJECT', text)
        self.assertNotIn('category-other', text)
        failed, calls, _, _ = self.invoke(records, env={'MANAGER_PROJECT': 'absent'})
        self.assertEqual(self.endpoints(calls), [])
        self.assertIn('NO_FRESH_MATCH', failed)

    def test_explicit_endpoint_mode_and_owner_override(self):
        text, calls, _, _ = self.invoke([], {('custom-host', '4242'): ([self.summary('legacy')], 'OK')},
            env={'MANAGER_HOST': 'custom-host', 'MANAGER_PORT': '4242', 'OWNER': 'other-owner'})
        self.assertEqual(self.endpoints(calls), [('custom-host', '4242')])
        self.assertIn('LEGACY_ENDPOINT', text)
        self.assertIn('owner: other-owner', text)
        for args, _ in calls:
            if args[0] in ('condor_q', 'condor_history'):
                self.assertIn('other-owner', args[args.index('-constraint') + 1])

    def test_automatic_owner_override_and_matching_direct_owner(self):
        records = [self.record('foreign-to-default', owner='other-owner'), self.record('default-owner', 1027)]
        text, calls, processes, _ = self.invoke(records,
            {('fixture-host', '1026'): ([self.summary('foreign-to-default', owner='other-owner')], 'OK')},
            env={'OWNER': 'other-owner'})
        self.assertEqual(self.endpoints(calls), [('fixture-host', '1026')])
        self.assertIn('owner_records=1 omitted_missing_or_other_owner=1', text)
        self.assertIn('category-foreign-to-default', text)
        self.assertEqual(processes, [])

    def test_empty_explicit_selectors_do_not_enable_automatic_mode(self):
        for key in ('MANAGER_PROJECT', 'MANAGER_HOST', 'MANAGER_PORT'):
            with self.subTest(key=key):
                text, calls, _, _ = self.invoke([self.record()], env={key: ''})
                self.assertEqual(self.endpoints(calls), [])
                self.assertNotIn('AUTO_OWNER_DISCOVERY', text)

    def test_explicit_log_freshness_and_manager_start_validation(self):
        now = time.time()
        for stamp, start, source in [(now, now - 60, 'STATS_LOG'),
                                     (now - 300, now - 600, 'CATALOG'),
                                     (now - 10, now, 'CATALOG')]:
            with self.subTest(source=source, stamp=stamp, start=start):
                row = {'project': 'selected', 'lastheardfrom': now, 'starttime': start}
                stats = {'timestamp': stamp * 1000000, 'tasks_waiting': 1, 'tasks_running': 1}
                output = io.StringIO()
                with mock.patch.object(dashboard, 'stats_tail', return_value=(stats, None, stamp)) as logs, \
                     contextlib.redirect_stdout(output):
                    dashboard.render_manager('fixture-host', '1026', 'selected', 'OK', [row],
                        (None, 'TIMEOUT', None, 'DISABLED'), 'OK', 120, 8, 'OK', '/explicit/logs')
                logs.assert_called_once_with('/explicit/logs/stats.log')
                self.assertIn('summary_source: ' + source, output.getvalue())
                self.assertNotIn('status: PROGRESSING', output.getvalue())
                if start > stamp:
                    self.assertIn('stats predates current manager', output.getvalue())


if __name__ == '__main__':
    unittest.main()
